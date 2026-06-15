# GA Counterexample Verifier (experimental, pluggable)

**Status**: Design — experimental direction, gated behind a strategy
interface. Default behaviour is unchanged; this is opt-in per experiment.
**Owners**: Synthex (search math), server (Synthex Hub, orchestration),
workers (rollout adapters).
**Theoretical grounding**: `CSHRL.Core.CoinductiveHomomorphism`
(`preserves`, successor-value dominance) and `CSHRLSynthesis` §10
(Online Synthesis / counterexample-guided refinement).

## TL;DR

CEGIS power lives in the **verifier** — the thing that finds
counterexamples. Today our verifier is passive: *fresh random seeds per
CEGAR round*. Most random initial states are easy, so most of the search
budget re-confirms the policy is fine where it already is.

This doc proposes replacing the verifier with an **active adversary** —
a quality-diversity (QD) genetic search that evolves *reachable* states
maximizing the policy's **regret** (its inverse-improvement margin) — and
feeding the resulting diverse counterexample batch into the existing
synthesizer. The synthesizer and the commit-gate soundness story are
untouched; only the *state-supply* step changes.

Hard requirement: this must be **pluggable and reversible**. The main
CEGIS/CEGAR story stays primary and load-bearing. If we dismiss the GA
direction, removal is deleting one strategy module plus one config enum
value — no changes to the default path, the protocol, or the data model.

## Problem statement

The hub controller drives each CEGAR step from a passively-sampled state
set and a passively-sampled seed set:

- `Mujoco.collect_states/2` — roll the *current* policy, record visited
  states. Passive: it samples the on-policy distribution.
- `Mujoco.seeds_for/3` — a deterministic block of fresh seeds per round.
  Passive: random initial conditions, no targeting.

Random testing is a weak verifier. On an env where the policy fails in a
narrow regime (a specific tilt/velocity band), random seeds rarely land
there, so the synthesizer is rarely shown the cases it most needs to fix.
The symptom is the Hopper result: the big coordinated gain only appeared
in a *late* CEGAR round, because earlier rounds were not concentrating
pressure on the policy's actual failure modes.

## What a counterexample *is* here

The optimality spec we approximate is `CoinductiveHomomorphism.preserves`:

    a ≤ₐ b  at s   ⟹   successor-value s a  ≤ₛ  successor-value s b

i.e. *if the policy ranks `b` over `a` at `s`, then `b`'s successor state
must have a pointwise-dominant optimal value stream.*

A **counterexample** is a reachable state `s` where the policy's induced
ranking violates this: the policy picks (ranks top) some action whose
successor is value-dominated by another action's successor. The magnitude
of that violation is the **regret**, which is exactly the
"inverse-improvement margin":

    regret(s) = max_b  V̂(next(s, b))  −  V̂(next(s, π(s)))

where `V̂` is the empirical return-to-go estimated by rollout (the same
machinery the scorer already runs). `regret(s) > 0` means "there exists a
locally better action than what the policy chose at `s`" — a concrete,
labelled counterexample: the state `s` plus its better action `b*`.

The verifier's job: **find reachable states maximizing `regret`.**

## Goals

1. **Pluggable / reversible (overriding constraint).** The verifier is a
   strategy chosen by config. `random` (today's behaviour) is the default
   and is bit-for-bit unchanged. The GA strategy is additive. Dismissing
   it is a one-module + one-enum deletion.
2. **Stronger counterexamples, faster convergence.** Reach target reward
   in fewer worker-seconds than the random-seed verifier by concentrating
   synthesis pressure on real failure modes.
3. **Soundness preserved (and ideally upgraded).** The commit-gate's
   strict same-seed monotonicity is untouched. The accumulated
   counterexample archive additionally enables a CEGIS termination
   argument stated directly against `preserves`.
4. **Architectural separation, unchanged.** Synthex drives search,
   server gates/coordinates, workers wrap rollouts. The GA runs on the
   finder side and emits *states*; it does not make workers fat.

## Non-goals

- **Replacing CEGAR.** CEGAR + the commit-gate remain the certifier and
  the primary algorithm. The GA verifier *feeds* it; it never writes the
  lineage directly.
- **A GA policy *proposer*.** That is a separate, lower-priority idea
  (see appendix). This doc is only about the verifier role.
- **Changing the persisted policy representation or the commit-gate.**

## Architecture: the strategy seam

### Verifier behaviour

Introduce a behaviour in Synthex (search side):

```elixir
defmodule Synthex.Verifier do
  @moduledoc "Supplies the states/seeds a CEGAR step refines against."

  @type ctx :: map()
  @type preds :: list()

  @doc """
  Given the current policy and context, return the state set + labels the
  synthesizer should refine against this step, plus the seeds used for
  same-seed scoring. `counterexamples` is the (state, better_action,
  regret) triples; `random` returns [] there.
  """
  @callback supply(preds, ctx, round :: pos_integer()) ::
              %{states: list(), seeds: [non_neg_integer()], counterexamples: list()}
end
```

Two implementations:

- `Synthex.Verifier.RandomSeeds` — **the default**. Wraps exactly
  today's `collect_states/2` + `seeds_for/3`. No behavioural change.
- `Synthex.Verifier.GAQD` — the experimental adversary (below).

### Where it plugs in

`Server.Workers.ExperimentController.run_step/2` currently does, inline:

```elixir
{states, _} = Mujoco.collect_states(preds_at_start, ctx)
features    = Mujoco.build_features(states, ctx)
seeds       = Mujoco.seeds_for(cegar_iter, 1, ctx)
```

This becomes a single dispatch through the configured verifier:

```elixir
%{states: states, seeds: seeds, counterexamples: cxs} =
  verifier(ctx).supply(preds_at_start, ctx, cegar_iter)

features = Mujoco.build_features(states, ctx)   # unchanged
# cxs is [] for :random; non-empty drives feature emphasis for :ga_qd
```

`verifier/1` reads one config key (below) and defaults to `RandomSeeds`.
Everything downstream — `build_features`, `optimize_bit`, the commit-gate,
the dashboard — is unchanged.

### Config knobs (lineage-forking)

Added to the experiment config, parsed in
`ExperimentBootstrap.build_context/3`:

- `verifier: "random" | "ga_qd"` — default `"random"`.
- `verifier_opts: { pop, generations, archive_bins, perturb_radius,
  regret_denoise_episodes }` — only read by `ga_qd`.

`config_sig` already forks lineages on any config-shape change, so a
`ga_qd` run is an isolated lineage and can be A/B'd against a `random`
run on the same env with zero risk to existing results.

### Worker protocol addition (additive, gated)

The GA verifier needs regret evaluation: from a batch of candidate
states, roll each admissible action one step then follow the policy, and
return per-state regret + best action. This is one new oracle command:

- `eval_regret` added to the `COMMANDS` map in `oracle_port.py`
  (and routed in `oracle_warp_port.py`), batchable like `score_bit`.

It is **only emitted when `verifier == "ga_qd"`**, so the default path
never sends it. Reuses the existing rollout code; no new env logic.
Workers that predate the command simply never receive it.

## The GA-QD verifier

Population = candidate **states**, not policies. Fitness = `regret(s)`.

- **Seeding (reachability).** Initialize the population from
  `collect_states` output — states the current policy actually reaches.
  This anchors the search to the reachable manifold so we don't
  synthesize against phantom off-distribution states (the adversarial-
  example trap).
- **Variation.** Bounded perturbation of state coordinates
  (`perturb_radius`), plus recombination of coordinate blocks. Stay
  within the env's valid state bounds; optionally project to the nearest
  reachable state via a short rollout-to-`s` check.
- **Quality-Diversity archive (MAP-Elites).** Bin state space into
  `archive_bins` cells (by a low-dim behavioural descriptor, e.g. coarse
  position/velocity bands). Keep the highest-regret state per cell. The
  output is a *diverse spread* of failure modes — avoids whack-a-mole
  where the synthesizer fixes one regime and the next round re-finds the
  same one.
- **Denoising.** Estimate `regret(s)` with `regret_denoise_episodes`
  rollouts and only admit `s` if regret exceeds a noise threshold — so we
  refine against real `preserves` violations, not variance (the lesson
  from the n_episodes=6 Hopper failure, applied to the verifier).

Output per step: the archive's elite states + their best-action labels →
fed to `build_features`/`optimize_bit` so the synthesizer concentrates on
exactly those states.

## Soundness: monotone archive ⇒ CEGIS termination

Keep a **permanent counterexample archive** (Hall of Fame) across rounds;
the synthesizer must continue to satisfy old counterexamples, not just the
current batch. This buys two things:

1. **No forgetting / no coevolution cycling.** The policy cannot regress
   on a previously-fixed failure mode, because that counterexample stays
   in the constraint set. (Standard fix for minimax/Red-Queen dynamics.)
2. **A convergence argument against `preserves`.** On a compact/finite
   reachable state set, counterexamples accumulate monotonically and the
   loop halts when the verifier can find no state with regret above the
   noise threshold. At that fixpoint the synthesized ranking satisfies
   `preserves` (within tolerance) on the *sampled reachable space* — a
   statement directly in the vocabulary of `CoinductiveHomomorphism`,
   strictly cleaner than the current reward-monotonicity story.

This is *additive* to the commit-gate; it does not replace it.

## Coevolution pathologies & mitigations

| risk | mitigation |
|------|-----------|
| Cycling / Red Queen | permanent counterexample archive (constraints accumulate) |
| Verifier collapses to one failure mode | QD/MAP-Elites archive enforces diversity |
| Off-manifold phantom counterexamples | seed from `collect_states`, bounded perturbation, reachability projection |
| Regret = noise | multi-seed denoising + admission threshold |
| Compute blowup | archive size caps population; one batched `eval_regret` per generation (warp-friendly) |

## Experimental plan & kill criteria

A/B `verifier: "ga_qd"` vs `verifier: "random"`, identical otherwise:

1. **Hopper-v5** — known target ~1100/ep. Metric: worker-seconds to
   reach 1000/ep. GA-QD should win by concentrating on failure regimes;
   if it does not beat random within noise, that is evidence to dismiss.
2. **Warp Ant** — the env where global/adversarial search should matter
   most. Metric: does `ga_qd` find *any* improving structure faster than
   random's stall.

**Kill criteria (decide to dismiss without sunk cost):** if `ga_qd` does
not beat `random` in worker-seconds-to-target on Hopper, and shows no
qualitative advantage on Ant, remove it. Because of the strategy seam,
removal = delete `Synthex.Verifier.GAQD`, the `eval_regret` handler, and
the `"ga_qd"` enum value. The default path, protocol, data model, and all
existing lineages are unaffected.

## Reversibility checklist (what "dismiss" touches)

- [ ] Delete `Synthex.Verifier.GAQD` module.
- [ ] Remove `"ga_qd"` from the `verifier` enum in `build_context`.
- [ ] Remove the `eval_regret` case from the oracle `COMMANDS` map.
- [ ] (Optional) drop the counterexample-archive table if added.

Nothing in `RandomSeeds`, `optimize_bit`, the commit-gate, or the
persisted policy schema changes at any point. The behaviour interface is
the only permanent addition, and it is independently useful (it makes the
state-supply step explicit and testable).

## Appendix: the GA *proposer* (separate, deferred)

A GP-over-PredProgs *policy proposer* (subtree crossover / typed mutation
over the serialized `feat/and/or/not` grammar, parsimony-pressured,
champion certified through the commit-gate as `engine: "ga+cegar"`) is a
distinct idea. It competes with CEGAR at a job CEGAR already does well
given budget, so it is lower priority than the verifier. It would reuse
the same `engine`-style config seam and the same certification handoff.
Documented here only to keep the two ideas from being conflated.

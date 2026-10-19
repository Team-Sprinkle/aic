# Autonomous execution prompt: transport, visual handoff, and insertion

Prepared: 2026-10-14. **These instructions take effect when the user asks a session to execute this file. Preparing this file did not start experiments.**

## Task and authority

Execute the full development plan in `docs/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md`, accounting for newer verified results. This invocation authorizes implementation, small diagnostics, simulation, data repair, training, evaluation, and documentation within the constraints below. It supersedes the earlier analysis-only restriction. Do not request routine confirmation between steps.

Work through both rule-based and learned transport, then the gated downstream work. A successful experiment is a reason to proceed to the next step, not to end the task. A failed experiment requires diagnosis and a targeted repair or a documented change of direction. Do not repeatedly run the same experiment with arbitrary parameter changes.

The objective is reliable autonomous transport to the **requested port**, a stable and correct visual handoff, and insertion from that handoff. A low teacher-forced action error or a momentary visit near a teacher endpoint does not achieve this objective.

Read this entire file before acting. Resolve uncertainty from code, artifacts and notes before asking the user. There is no blanket one-hour root-cause limit. Use bounded individual experiments and evidence to decide extensions. Do not claim success to escape a failed gate.

## First: resume safely and establish current state

1. Read applicable `AGENTS.md` instructions, `docs/LOCAL_WORKFLOW.md`, and the rootless Docker guide referenced there. Check the checkout and symlinks; similarly named workspace directories may be different checkouts.
2. Read these documents in order:
   - `docs/STATUS.md`
   - `docs/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md`
   - `docs/experiments/2026-10-13-transport-independent-root-cause-review.md` and its machine-readable companion
   - `docs/answers/2026-10-14-rule-based-transport-and-final-plan.md`
   - `docs/answers/2026-10-14-waypoint-variation-grouping.md`
   - `docs/ACTIVE_TRANSPORT_DATASET.json`
   - Relevant newer experiment records, `docs/SELECTED_PORT_HANDOFF_PLAN.md`, and the existing recovery/SERL plan before reaching that phase.
3. Inspect git status, active training/evaluation processes, rootless containers, GPU ownership and memory, disk space, existing experiment manifests and completion markers. Let active jobs finish. Do not duplicate jobs or stop another session's work. Use existing completed results when they answer the exact question.
4. Create a durable execution record under `docs/experiments/` and a corresponding JSON state file. Record each stage/branch, hypothesis, prerequisites, current status, budget, gate, artifacts, commands, outcome and exact next action. Update them after each material result.
5. Resolve actual dataset/checkpoint paths from manifests. Do not guess a similarly named checkpoint is the selected one. Record hashes/configurations for frozen comparisons.

Use **one GPU initially**, and never more than **four project GPUs concurrently**, counting existing project jobs. Respect other users. Use rootless Docker without sudo. Check available quota before creating containers; reuse a suitable running container when supported. Store large new artifacts on a disk with verified free space, preferably the established NAS archive location; `/data1` was nearly full. Do not delete original data or failed results to make room without following the existing retention policy.

## Facts to verify and preserve

These are findings from the October 13–14 audit, not assumptions to overwrite silently if newer evidence differs:

- The selected SC source dataset has **104 successful full episodes: 83 train / 21 validation**, from 200 collection attempts. The active union also contains SFP data; report its actual manifest counts separately. Do not describe 104 source insertions as autonomous actor successes.
- SC teacher transport prefixes end at about **−168 mm TCP axial / −148 mm plug-tip axial**. None of these prefixes reaches the old −100 to −70 mm TCP gate. Later teacher approach actions exist but were excluded. This component-contract mismatch must be resolved before another broad training run.
- Fine70 visited within 20 mm / 5 degrees of its taught endpoint on 5/6 reused development scenes, but often drifted away. This was retrospective proximity, not demonstrated visual handoff. The latest diffusion runtime sets handoff probability to zero; it does not inherit a trained stop head from an older model.
- Complete-12-slot filtering removes final-return examples from 29/83 training episodes and removes all 61 cached handoff-positive rows from the valid-action subset. Repeated command indices can be valid held commands. Use source timestamps and causal command semantics rather than requiring command indices to advance every tick.
- Preserve raw observations and command timing. Pause cleanup should change training-anchor selection/weights, not erase real elapsed time from histories or invent future commands. No post-reset geometry in terminal labels.
- The actor predicts complete future targets relative to the measured decision-time TCP anchor. Verify this exact contract end to end. Do not revive a residual/delta adapter on top of a frozen actor, or switch the transport action frame to an absolute global frame. Local body-frame motion and port-frame insertion are distinct contracts.
- Fine70 and overfit results are promising comparisons, but extra updates confound claims about history alone. Teacher-forced submillimeter error on sampled frames is not autonomous mastery.
- The current visual baseline uses frozen globally pooled ResNet features. Prior DINO linear probes did not establish a temporal-policy advantage or disadvantage. Spatial ResNet and DINOv2-S patch features remain conditional comparisons.
- SC teachers use the **Low Lateral Bypass (LLB)** route. Rejected over-card SC routes must remain excluded. Keep SFP teachers consistent with the successful SFP route.
- The reported **5–6 cm waypoint spread is across randomized scenes, within each port/stage separately**. It is not repeatability in one fixed setting or tracking error. Decompose board, rail and robot-start variation before selecting margins.
- Successful LLB teacher retargeting used true board/rail geometry. It is an oracle data-generation method, not an observation-only deployment solution. Existing SC orientation and rail coverage is narrower than the released settings.

If a finding has changed, record the new evidence and date instead of silently retaining stale numbers.

## Invariants for every experiment

- Keep all reserved final scenes sealed. Use episode/reset-grouped train, validation and development partitions. No transition-random split as evidence of generalization. Reused diagnostic scenes and fresh development scenes must be reported separately.
- Simulator geometry, exact target transforms and configuration files are allowed for teacher generation, labels and diagnostic scoring. They must not select autonomous routes/crops, authorize stops, or condition the deployable actor. Audit the runtime input path explicitly.
- Public task identity, measured robot state and camera/history observations are permitted. Requested-port perception must distinguish neighboring ports. A visible wrong port is not a positive example.
- Harmless cable/card touch is acceptable. Sustained snag, harmful blockage and prohibited contacts are failures. Do not restore an obsolete zero-contact gate.
- Record commanded and measured motion, force/contact, stage transitions, stop proposals, visual decisions and terminal observation before reset. Report TCP and plug-tip measurements separately.
- Freeze thresholds, splits, budgets and primary metrics before each comparison. If the component contract changes, version it and rerun matched controls. Never loosen a failed gate retrospectively and present the same run as a pass.
- Preserve original datasets. Put repaired/cleaned data in versioned copies with provenance and exclusion reasons. Keep unsuccessful trials and checkpoints needed for comparisons.
- No world-model/SEER/imagination work. No RL as a substitute for diagnosing a broken transport executor, task encoding, or perception contract.
- The user temporarily relaxed inference latency to establish route correctness. Still measure it; restore the 300 ms p95 deployment target later. Simulator task timeouts remain relevant throughout.

## Execution stages

### 1. Establish the actual transport-to-insertion contract

Run current coarse perception and refinement on recorded teacher/actor endpoint windows. Determine where the requested port can actually be identified and localized, and establish the insertion actor's demonstrated starting position/orientation envelope. If that envelope has not been validated, report it as unknown and run a bounded diagnostic instead of assuming it.

Assign ownership of the missing approach: extend transport using existing safe teacher approach actions, or validate a downstream approach that genuinely handles the old endpoint. Define proposal, hold, correct-port approval, pose refinement, insertion acceptance, timeout and rejection behavior explicitly. A coordinate prior may reject implausible proposals but cannot alone approve insertion.

Initial budget: 2–4 hours of offline analysis. Gate: consistent labels, controller transitions and evaluation targets. Save representative endpoint images with clearly distinguished predictions and diagnostic truth.

### 2. Screen rule-based SC transport

Using training scenes and failed collection attempts, first test whether a conservative common corridor exists. Separate variation attributable to board transform, selected rail and robot start. An oracle decomposition is an upper-bound analysis, not a deployable observation source.

Preserve the demonstrated sequence: back away → move left and descend in the clear corridor → travel beside and beyond the row → move right beyond the cards → tilt with tool clearance → return toward the requested port → hold for visual acquisition.

Check measured tool/plug height and swept gripper volume, not just TCP position. Advance stages by measured completion, not fixed wall-clock durations. Add progress/force limits and explicit failure handling.

Evaluate candidates in this order:

1. Conservative corridor with connection from measured robot start.
2. Observation-adapted template using rough board/row extent and orientation, if a fixed corridor is insufficient.
3. Small observation-selected template library if route topology must differ.
4. Rule prefix plus learned/visual final approach if only the final return remains weak.

Do not use a scene-ID lookup or exact simulator geometry in autonomous planning. Include both ports, one/five-card extremes, start/rail variation, and a rotated-board development case as an explicit support test. MoveIt can check rigid reachability/clearance; it does not establish cable safety.

Initial budget: 2–4 hours offline screening. Execute alternatives when their prerequisite failure occurs; otherwise record why they are unnecessary. Do not train additional template selectors without evidence that the simpler option fails.

### 3. Run a bounded Gazebo rule-versus-learned comparison

Freeze the candidate, then run six nonfinal SC settings with shared downstream checks. Use matched frozen learned traces where valid, otherwise rerun them. Label privileged teacher controls separately.

Save wrist plus overhead/side videos, preferably Ubuntu-compatible H.264 at 10–20 fps, and review the critical motion intervals. Diagnostic cameras are not policy inputs. Preserve footage before reset; do not treat asset removal after episode termination as route failure.

Gate: **at least 5/6 stable arrivals in the validated visual acquisition region**, no wrong-port accepted handoff, and no sustained blockage/prohibited contact. This is a feasibility screen, not a reliability claim. Test at most three promising edge cases with ±10/20 mm perturbations initially.

For failure, classify scene-estimation error, unreachable corridor, cable load, tilt collision, final-return error, or visual rejection. Test one distinct repair per identified cause before reviewing the design. If multiple approaches fail for the same underlying issue, address that issue rather than expanding a margin sweep. Continue the learned branch even if the rule candidate fails.

### 4. Repair learned-actor labels and stopping supervision

Rebuild targets from original timestamped commands with the exact controller's causal hold semantics and bounded command age. Preserve episode/reset boundaries, use per-slot validity for incomplete terminal horizons, and retain stop examples independently of future-action validity.

Audit at least 12 representative windows spanning held commands, cleanup gaps, tilt, final return, approach and terminal/reset boundaries. Audit all episode-level counts, especially final-stage survival. Verify stage labels against observed progress rather than nominal plan times.

Create a versioned dataset and publish counts by task, port, card count, stage and split. Gate: every retained label has valid provenance and terminal examples survive, or have a documented exclusion reason. Do not collect a large replacement dataset to avoid fixing this known bug.

### 5. Establish executor parity before retraining

Replay recorded teacher targets through the exact actor chunk executor on up to three nonfinal starts, alongside original-teacher controls: at most six initial runs. Check anchor transforms, action timing, gains, target age, chunk boundary behavior and stopping.

Proposed diagnostic gate: reproduce the taught endpoint within **10 mm / 3 degrees** without harmful blockage. Declare this tolerance before running; explain any justified prospective change. This is an oracle execution test, not autonomous policy success.

If it fails, fix the executor first. Distinguish inference hold gaps from wrong targets and label loss. Test buffering/asynchronous execution only after demonstrating a timing-related problem; reanchor outstanding targets correctly.

### 6. Run the matched learned-transport experiment

Retain frozen fine70. Compare old versus repaired supervision under the same architecture, episode split and update budget. Include supervised proposal/stop heads in both arms with a documented common label definition; keep the frozen original as a separate reference. If action repair and stopping changes cannot be isolated in two arms, add the smallest ablation needed to identify the cause.

Start with one GPU and an initial **four GPU-hours per arm**. This is an initial allocation, not proof of convergence. Predeclare a plateau rule using repeated held-out evaluations, phase-specific action/orientation errors and stop quality. If still improving, extend matched arms in documented increments. Report budget-limited runs honestly. Do not repeatedly inspect live scenes while tuning and then call those scenes held out.

Evaluate reused diagnostic starts, then freeze selection and use six fresh grouped SC development starts. Gate: at least 5/6 stable correct acquisition arrivals, no observed wrong-port approval or sustained blockage, and clear improvement over the matched control. Compare the selected learned result against the rule candidate under identical downstream checks.

Report ordered route progress/deviation, endpoint translation/orientation, stable stopping, coarse approval, insertion eligibility and full insertion separately. Closest-point route distance alone can hide backwards travel and must not be the only measure.

### 7. Explore the remaining model directions only against measured failures

Track every branch below in the ledger. A branch may be ruled out with evidence or marked conditional when its prerequisite is absent; do not quietly omit it or claim an unrun experiment succeeded.

| Remaining failure | Comparison to execute | Requirement to continue |
| --- | --- | --- |
| Poor visual route/target discrimination | Spatial ResNet features versus DINOv2-S patch features; matched policy head, data and budget | Held-out task-relevant improvement plus live benefit |
| Ambiguous route stage/history | Dense recent context plus distance/rotation-spaced older memory; choose spacing from measured routes | Better final-return/stopping behavior, not just offline loss |
| Route progress confusion | Observed progress/stage supervision with validated labels | Fewer wrong-stage actions on grouped held-out episodes |
| Synchronous inference gaps | Exact-executor timing control versus correctly reanchored buffering | Measured reduction in tracking/route error |
| Off-demonstration failures | At most three causal teacher-continuation pilots from actual actor states | At least two valid safe continuations before scaling |
| Capacity/optimization remains limiting after repairs | One adequately trained capacity or optimization comparison, retaining intentional-fit and validation-selected checkpoints | Evidence of underfitting first; improvement beyond extra-update effects |

Use an initial four GPU-hours per training arm and six matched development episodes per live comparison unless the measured hypothesis needs a smaller diagnostic. Extend only with documented rationale. Do not blindly sweep architectures or resurrect a residual action adapter. Avoid conflating longer history, longer action horizon, timestep and extra updates; isolate changes.

### 8. Broaden supported settings and integrate insertion

Compare supported performance for both SFP-to-NIC and SC-to-SC, both requested SC ports, card counts, broad robot starts, rails and board orientations. Use published evaluation distributions to design development coverage without opening reserved final scenes or supplying exact scene parameters to the policy.

Repair missing routes before collecting many examples. For SC, never reintroduce rejected over-card teachers. For SFP, retain its intended teacher logic. Allow different transport methods per family if evidence supports that simpler solution.

Integrate: transport proposal → hold → requested-port coarse approval/crop → original-resolution pose refinement → insertion actor starting-envelope check → insertion. Train or repair perception/insertion only for an identified coverage or interface gap. Use predicted perception outputs for deployable insertion; simulator labels remain supervision/scoring only. Audit task encoding and coordinate transforms across the switch.

If insertion needs retraining, use a verified, sufficiently diverse near-port dataset covering the measured transport endpoint distribution with buffer. Compare an appropriate warm start and scratch initialization only where existing evidence is insufficient. Do not mistake success with oracle poses for success with predicted poses.

Before each expansion, declare the number and grouping of episodes, thresholds and continuation rule. Initial feasibility screens can use six episodes; a reliability conclusion needs a broader frozen development evaluation with uncertainty and all failures reported. Measure full-path latency p50/p95/p99, wall time and reset/terminal behavior.

### 9. Gated recovery, RL and Gazebo transfer

Proceed only after ordinary autonomous supervised transport, correct visual handoff and insertion pass their declared development gates. If they do not pass, continue targeted repairs; do not hide the gap with privileged guidance or RL.

Read the current approved recovery/SERL records and reconcile architecture changes before implementing this phase. Keep one shared learned policy for normal insertion and recovery across task families where that is the approved contract; rule-based transport is an evaluated alternative, not permission to silently multiply recovery policies.

Compare **no backoff**, **local backoff**, and **hierarchical backoff** under matched starts, policy checkpoints and interaction budgets. Keep hardcoded supervisor actions explicitly marked in replay. Track prior measured poses, retreat levels, force/progress, retries and lateral exploration. Do not train an actor to imitate privileged supervisor decisions as if they were autonomous predictions.

Use reproducible Isaac failures for efficient SERL-style training, with a bounded comparison of the previously approved offline warm start versus direct online continuation if not already resolved. Set interaction/update budgets, reward audits and stopping rules before launch; prevent reward improvement through non-insertion shortcuts. Validate observation/camera/action parity, then transfer and evaluate in Gazebo. Gazebo is the required final environment; Isaac success alone is insufficient.

Keep world models, SEER and imagination disabled. Reserved final evaluation remains separately authorized: prepare a frozen evaluation package, but **do not open final scenes under this prompt**.

## Persistence, reporting and completion

- Maintain a table for every stage and conditional branch: `not_started`, `running`, `passed`, `failed`, `blocked`, or `not_applicable_with_evidence`. Preserve failed attempts and alternative checkpoints.
- At a failure: inspect traces and critical video frames; state a specific cause or uncertainty; design a falsifiable targeted repair; compare against the frozen control; record whether the gate passed. After repeated evidence against a design, revise the design and continue another justified branch. Do not spend unlimited compute repeating an exhausted hypothesis.
- During training, check completion and failures periodically. Avoid repeatedly reading full logs or producing lengthy commentary about tiny loss changes. While waiting, do useful independent audits within the resource limit. Launching a long job does not complete the task.
- After each stage, update the experiment record, JSON state, `docs/STATUS.md`, `docs/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md`, and relevant artifact map. Update `docs/EXPERIMENTS.md` and `docs/SELECTED_PORT_HANDOFF_PLAN.md` when conclusions change. Use existing main documents; avoid competing status files with inconsistent claims.
- Every run needs exact command/config, dataset/split hashes, checkpoint identity, seed/reset grouping, runtime input provenance, metrics, diagnostic images/video paths and decision. Explain conclusions in plain language. Save substantive user-facing responses under `docs/answers/` and link them.
- On interruption or context restart, read the ledger and inspect processes before resuming. Save the next runnable action. Do not duplicate a still-running job.
- Do not stop for routine confirmation. Stop dependent work only for a genuine external blocker, missing user-only information/credentials, unsafe resource conditions, or the sealed-final authorization boundary. Continue independent branches if possible. State the concrete blocker and completed evidence instead of asking a generic “shall I continue?”
- Completion means the development plan has been worked through with evidence for each branch, the selected system has passed its declared gates, and the remaining final-evaluation package is ready; or there is an explicit evidence-backed blocking conclusion after distinct repairs have been exhausted. A failed gate stays failed/open. Do not claim that every optional idea was trained if it was instead ruled out.

Begin by checking current files and active jobs, then write the execution ledger and carry out the first unresolved stage.

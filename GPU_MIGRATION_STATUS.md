# GPU Migration Status

Last updated: 2026-08-11

## Current phase

Phase II native rigid-mesh collision migration — implementation and required real-
object measurements complete; documentation and full regression verification in
progress

## Phase II checkpoint: pre-implementation audit

- Read `GPU_MIGRATION.md`, this status file, and
  `generated/gpu_validation/validation_report.md`; inspected the dirty worktree and
  preserved all existing GPU-migration changes.
- Inspected the real `n0500_a0p3_b0p6` large/high/high/high generated XML. Its
  rigid branch has one ordinary free joint, explicit mass/inertia, and
  `flexcomp type="gmsh" dim="3" dof="trilinear" rigid="true"`; it has no
  elasticity element.
- Compiled that XML with MuJoCo 3.11.0. The whole model has `nq=38`, `nv=36`,
  `nflex=1`, `nflexvert=2387`, `nflexedge=12602`, and 8552 tetrahedral flex
  elements. All 2387 vertices reference object body 28, `nflexnode=0`, and the
  flex adds no generalized coordinates. The object contributes only its free
  joint's 7 qpos/6 qvel values.
- Confirmed that collision uses the compiled flex shell (3328 exterior triangles)
  and exposes geom/flex/element/vertex contact identifiers. Compiled rigid-flex
  contact defaults are friction `1 0.005 0.0001`, `condim=3`,
  `solref=0.02 1`, `solimp=0.9 0.95 0.001 0.5 2`, zero margin/gap,
  contype/conaffinity `1/1`, and priority 0.
- Reconfirmed the recorded baseline: free motion matches (`qpos=1.61e-8`,
  `qvel=5.12e-7`, touch 0), settled rigid-flex contact diverges
  (`qpos=0.00161`, `qvel=0.804`, touch `74.35`), and the native rigid control
  remains tight (`5.07e-8`, `3.12e-6`, `1.48e-5`).
- Reproduced the separate stock MJWarp 3.11 rigid-flex zero-sized edge-Jacobian
  illegal CUDA access (`nJfe=0`). Earlier numeric rigid-flex measurements used the
  documented diagnostic-only invariant-edge workaround; production still rejects
  flex models.
- Inspected the representative source as little-endian binary GMSH 2.2 containing
  2387 nodes and 8552 first-order tetrahedra, with no explicit surface elements.
  Raw bbox dimensions are `2.17316958 1.96864511 3.86545609`; the existing
  `0.03125` scale gives approximately `0.06791155 0.06152016 0.12079550` metres.
- Recorded the implementation/test design in the session before making code
  changes. Surface extraction will retain once-occurring oriented tetrahedron
  faces, preserve raw coordinates, validate topology/geometry, and cache by source
  hash plus converter version. The native geom will preserve the free joint,
  body pose, explicit inertia, names, and compiled contact defaults. Flex radius
  has no exact native-geom analogue and is an explicit old/new scientific risk.

Exact next task: implement and test deterministic GMSH exterior-surface conversion,
cache reuse, geometry diagnostics, and rigid-representation manifests before
changing generated MJCF.

## Phase II checkpoint: conversion, native MJCF, and initial parity

- Added `object_conversion/gmsh_to_rigid_surface.py`. It parses actual GMSH 2.x
  ASCII/binary sources, extracts once-occurring tetrahedron faces with outward
  winding, removes internal faces, compacts boundary vertices, emits deterministic
  OBJ, validates bbox/centroid/area/volume/components/watertightness/winding, and
  caches by source SHA-256 + converter version + conversion parameters.
- The representative real source converts from 2387 source vertices/8552 tetrahedra
  to 1666 boundary vertices/3328 triangles. It is one watertight component with no
  boundary/non-manifold/winding-error edges, exact bbox, `2.16e-15` relative volume
  error, and `3.53e-16` volume-centroid L2 error.
- Converted all 24 `sphere_study_v1` sources. All are one-component/watertight,
  all 24 hashes are distinct, and output topology ranges from 3184-4832 triangles
  and 1594-2418 boundary vertices. Scaled geometry retains the size, aspect,
  macro, and roughness object distinctions.
- Rigid custom generation now emits `<mesh name="custom_object_mesh" ...>` plus
  `<geom name="object" type="mesh" ...>` and no manipulated-object flex/flexcomp.
  Deformable generation remains the existing `flexcomp rigid="false"` path.
- Added `rigid_object_representation.json` with source/conversion hashes, cache key,
  raw/scaled geometry, mass/inertia/body pose, contact mapping, and unmapped rigid-
  flex radius semantics. N=500 and N=1000 generation reuse the identical cache key
  and OBJ basename.
- The real N=500 new model loads in MuJoCo 3.3.1 and 3.11.0 with `nflex=0`,
  `nq=38`, `nv=36`, 500 touch channels, one conventional mesh geom, and unchanged
  object free joint, mass `0.976562`, center of mass, diagonal inertia
  `0.00305176 0.00305176 0.00305176`, friction/solver/filter parameters.
- Capability detection now classifies compiled structure as `rigid_mesh_geom`,
  `native_rigid_geom`, `rigid_flex`, or `deformable_flex`; GPU support is granted
  to rigid geom models and still fails closed for either flex class.
- Added rich contact/tactile diagnostics (contact IDs, positions, normals,
  distances, forces; object state; tactile max/mean/median/RMSE/correlation,
  activation overlap, totals, and sensor/site/region mapping) plus durable CPU
  old/new and CPU/MJWarp diagnostic CLIs.
- Initial exact-model CPU/MJWarp N=500 results:
  - free motion: qpos `1.61e-8`, qvel `5.12e-7`, tactile 0;
  - one step after 200 CPU settling steps: qpos `6.26e-7`, qvel `4.45e-4`,
    tactile max `0.03470`, mean `6.94e-5`, RMSE `0.00155`, active Jaccard `1.0`,
    correlation ~`1.0`, total tactile magnitude relative error `0.00876`;
  - all three contacts use the same geom pairs after order-independent matching;
  - a full 20-substep policy-action test passed state/tactile tolerances and exact
    sparse reward/success parity.
- The backend discrepancy improves dramatically versus old rigid flex (touch max
  error `74.35` -> `0.03470`), supporting collision representation as the primary
  source. It is not as tight as the native-block control after contact and remains
  an Outcome-B precision caveat over longer rollouts.
- CPU old-flex versus new-mesh is materially different and must not be hidden:
  from the same old-flex settled state, old/new have 89/3 contacts, one-step qvel
  differs by `1.0404`, old tactile total is `435.19` while new is zero for that
  exact state, and independently settled trajectories diverge. Mass/inertia remain
  exactly equal. New mesh experiments therefore require distinct provenance and
  cannot be mixed with historical rigid-flex CPU results.
- Focused CPU suite: 20 tests passed. Custom rigid GPU task policy-step test passed.
  Converter/generator/parity modules compile and `git diff --check` is clean.

Exact next task: finish controlled contact-onset fixtures and order-independent
contact metrics, run real N=500 learning smoke/complete-loop benchmark, then repeat
the observation/training/overflow checks at N=1000 before updating final docs.

## Phase II checkpoint: controlled localization, learning, and throughput

- Added `shadowhand_gpu/rigid_flex_diagnostic.py` and `diagnose_rigid_flex.py` with
  controlled A-E fixtures: free/no-contact, one approaching hand geom, isolated
  fingertip, isolated shallow palm, and settled contact. The diagnostic-only MJWarp
  edge workaround validates that all flex vertices belong to one rigid body and
  refuses every deformable/multi-body flex; the production backend remains unchanged
  and rejects flex.
- Imported initial CPU rigid-flex contacts agree with Warp at about `3e-8` m, but
  after Warp's first collision pass it produces 2674 flex-flex contacts even with
  the object at z=2 and no CPU contacts. They identify `flex=(0,0)` and
  `geom=(-1,-1)` despite `selfcollide="none"`. The approach fixture diverges at
  Warp step 1; CPU's intended fingertip contact begins at step 5. In the settled
  comparison one Warp step changes 89 matched contacts to 2757. This directly
  localizes the old failure to collision identification.
- Re-ran new-mesh matched-state parity with order-independent contact pairing. Free
  motion remains qpos `1.61e-8`, qvel `5.12e-7`, touch zero. After 200 CPU settling
  steps, one matched step has qpos `6.26e-7`, qvel `4.45e-4`, and touch max/mean/
  median/RMSE `0.03470 / 6.94e-5 / 0 / 0.00155`. Both backends activate the same
  sensor (Jaccard 1.0), total magnitude differs by 0.876%, and all three contact
  geom pairs match. Contact position/normal/distance/force max errors are
  `1.33e-4 m / 0.00291 / 1.95e-7 m / 0.05365`.
- Defined the settled execution gate as qpos <=`1e-5`, qvel <=`1e-3`, touch max
  <=`0.05`, total-touch relative error <=2%, active Jaccard >=0.9, and exact contact
  count/geom pairs. The 20-substep action gate is qpos <=`3e-4`, qvel <=`0.02`,
  touch <=`0.2`, and exact reward/success. Both pass. These do not claim long-rollout
  equivalence.
- The result is Outcome B: changing from rigid-flex to conventional rigid mesh
  eliminates the dominant error (`74.35` -> `0.03470`) but remains looser than the
  native-block control (`1.48e-5`). CPU old/new collision fields are themselves
  materially different, so new `rigid_mesh_geom` results cannot be pooled with
  historical rigid-flex results.
- Real custom N=500 learning smoke: 64 worlds, 12032 transitions, 64 completed
  episodes, 693 unchanged TQC updates, three evaluations, metrics JSON, two periodic
  checkpoints, and a final checkpoint in 37.48 seconds. The `[0,0,0]` success curve
  is plumbing evidence only.
- Real custom N=500 complete-loop throughput at 32/64/128/256 worlds is
  `181.21 / 180.57 / 195.21 / 192.11` transitions/s with the reference update
  ratio. The 128-world result uses 21 updates/step, reaches 32.03 updates/s and
  3904.17 physics world-steps/s, needs 512.27 seconds/100k, peaks at 360,539,136
  PyTorch bytes, and has zero overflow. The same new XML on the six-env CPU loop is
  75.24 transitions/s, giving a real-object matched-model speedup of 2.59x.
- Real custom N=1000 has exactly 1000 touch values, raw observation width 1062,
  policy input 1076, and a 19968-transition/173.9 MB replay allocation. Its 12032-
  transition smoke also completes 64 episodes and 693 updates with metrics and
  checkpoints. The 128-world complete loop is 160.24 transitions/s, 26.29 updates/s,
  and 3204.73 physics world-steps/s with 75 maximum constraints/world and no
  overflow.
- Added a real 24-object conversion regression and a GPU N=500/N=1000 observation,
  replay-allocation, load/step, and capacity test. All 24 objects retain distinct
  geometry and pass format/topology validation. Updated GPBO/study validation to
  allow rigid `msh_file` jobs while keeping flex/deformable fail-closed.
- Added durable machine-readable measurements at
  `generated/gpu_validation/phase2_rigid_mesh_summary.json` and extended the main
  validation report without removing Phase-I results.

Exact next task: run the complete CPU and opt-in GPU suites, correct any regression,
then mark the checkpoint complete with final test counts and remaining long-horizon
scientific limitations.

## Completed work

- Read the migration request and inspected the repository before making changes.
- Inspected the CPU trainer, dynamic environment, sensor/XML generation pipeline,
  study/GPBO orchestration, generated assets, existing tests, and installed package
  versions.
- Inspected the installed SB3-Contrib 2.7.1 TQC and HER implementations to record the
  exact target/loss/sampling behavior that the CUDA path must preserve.
- Compiled and inspected a real N=500 generated model. It has `nq=38`, `nv=36`,
  `nu=20`, `nflex=1`, 529 scalar sensor values, and 500 touch sensors.
- Recorded the direct MuJoCo Warp architecture, interoperability plan, replay memory
  estimate, parity gates, benchmark protocol, risks, and milestones in
  `GPU_MIGRATION.md`.
- Confirmed PyTorch sees an NVIDIA GeForce RTX 3050 Laptop GPU with approximately
  3.68 GiB, even though `nvidia-smi` currently reports an NVML library mismatch.
- Ran the complete existing repository test suite successfully.
- Created an isolated temporary GPU environment with MuJoCo/MuJoCo Warp 3.11.0
  and Warp 1.16.0. The CPU reference Conda environment remains unchanged.
- Added a version-aware generated-XML loader. MuJoCo 3.11 removes the obsolete
  `option.apirate` attribute in memory, while MuJoCo 3.3 loads the original file.
- Explicitly disabled the new MULTICCD and NATIVECCD defaults under MuJoCo 3.11 to
  match the effective collision settings of the MuJoCo 3.3 reference model.
- Added a compiled sensor layout using `sensor_adr`/`sensor_dim`. On the actual N=500
  model, the 500 touch values are the contiguous `sensordata[29:529]` span.
- Added a direct batched MJWarp backend with cached zero-copy PyTorch views for qpos,
  qvel, controls, complete sensor data, and touch data on CUDA.
- Diagnosed and guarded a MuJoCo Warp 3.11.0 rigid-flex bug: its private flex-edge
  kernel writes into a zero-sized Jacobian for two vertices attached to the same
  body. The project workaround computes invariant edge length/zero relative velocity
  only for validated single-body rigid flexes; other flex models fail closed.
- Stepped the actual generated N=500 `nflex=1` model successfully on CUDA with 8192
  contact slots and 4096 constraint slots per world.
- Added a capability-report CLI, raw simulation benchmark, GPU requirements file,
  and CPU-safe/opt-in GPU tests.
- Reproduced the study's N=1000, alpha=0.1, beta=0.9 candidate in a temporary
  directory using the unchanged generator. It realized the expected existing
  area-weighted allocation 74/272/654.
- Loaded and stepped that N=1000 model on CUDA. It has 1029 scalar sensor values,
  a contiguous 1000-value touch span at `sensordata[29:1029]`, and its initial state
  used about 2471 contact candidates and 2510 constraints.
- Added CUDA state transfer for qpos, qvel, controls, time, and acceleration warmstart
  while keeping cached PyTorch/Warp views device-resident.
- Added a matched-state CPU MuJoCo 3.11 versus MJWarp one-step parity harness with
  no-contact and CPU-settled contact modes.
- Proved free/no-contact dynamics parity on N=500: qpos max absolute error
  `1.61e-8`, qvel `5.12e-7`, and tactile error zero.
- Generated a valid native rigid-box N=90 control fixture in a temporary directory.
  Its settled-contact parity also passes tightly: qpos max error `5.07e-8`, qvel
  `3.12e-6`, and touch max error `1.48e-5`.
- Demonstrated that the generated rigid-flex object's contact path does not have
  acceptable parity: after the same 200-step CPU settling used by task reset, one
  compared step has qpos max error `0.00161`, qvel `0.804`, and touch `74.35`.
- Repeated the contact comparison using MJWarp's full `put_data` import; results were
  unchanged, ruling out omitted state-transfer fields.
- Evaluated and rejected an in-memory tetrahedral-boundary-to-rigid-mesh conversion.
  The original CPU flex produced 89 contacts versus 3 for the mesh, qvel changed by
  up to `1.04`, and the original peak touch force disappeared. This is not a valid
  scientific-equivalence workaround.
- Applied the explicit 2026-08-10 scope decision to skip flex implementation. The
  production backend now rejects every compiled model with `nflex > 0` before MJWarp
  allocation and reports its support mode as `native_rigid_only`.
- Removed the private MJWarp flex-edge workaround from the production codebase.
  Existing flex parity and throughput results remain recorded diagnostics only.
- Added a checked-in native-rigid tactile fixture and moved backend stepping/state
  transfer tests to it. The real generated N=500 flex model is now a fail-closed test.
- Added a pure-PyTorch squashed Gaussian actor, quantile-critic ensemble, truncated
  entropy-adjusted Bellman target, quantile-Huber loss, learned entropy coefficient,
  Adam update orchestration, Polyak target updates, and resumable learner state.
- Matched SB3-Contrib 2.7.1 actor/critic forward values, sampled action log
  probabilities, parameter gradients, quantile-Huber loss/gradients, target formula,
  and a complete entropy/critic/actor/target optimizer step under fixed parameters.
- Kept all per-update metrics as CUDA tensors, avoiding hidden scalar transfers and
  synchronizations in the learning hot path.
- Added a standalone CUDA TQC benchmark and validated the full project batch size
  and network architecture for both N=500 and N=1000 observation widths.
- Added float64-state CUDA running moments and Dict observation/discounted-return
  normalization matching SB3 VecNormalize's update order, clipping, reset behavior,
  and current-stat replay sampling semantics.
- Preserved Gym Dict/SB3 CombinedExtractor policy order exactly as
  `achieved_goal`, `desired_goal`, `observation`.
- Added a memory-planned CUDA HER ring buffer. It stores raw transitions, samples
  only complete episodes, invalidates an entire old episode before overwrite,
  selects inclusive future goals from future `next achieved_goal`, recomputes sparse
  ShadowHand rewards on device, masks timeout dones, and emits real samples before
  virtual samples exactly like SB3.
- Added explicit replay preflight behavior: requested capacity fails with required
  and budget bytes, while opt-in auto-capacity records the selected vector-aligned
  capacity. No partial allocation occurs before the check.
- Proved fixed-index episode metadata and HER output parity directly against
  SB3 `HerReplayBuffer`, and verified normalization/replay/TQC batches stay on CUDA.
- Fixed CUDA device resolution found during verification: bare `device="cuda"` now
  resolves to the current device index before calling PyTorch memory APIs.
- Added a standalone benchmark for CUDA HER sampling, normalization, reward
  recomputation, and SB3-order policy flattening.
- Added a native-rigid CUDA ShadowHand task over the direct MJWarp backend. It owns
  per-world actions, goals, episode counters, independent RNG state, observations,
  sparse rewards, success/timeout signals, and masked reset orchestration.
- Matched the reference action-conditioning order and absolute actuator mapping:
  input clip, scale, optional clip, exponential smoothing, final clip, then actuator
  control-range conversion.
- Matched the task observation contract: 24 hand positions, 24 hand velocities, six
  object velocities, seven achieved-goal values, N tactile values, and the time
  feature. The policy order remains achieved goal, desired goal, observation.
- Added device-side per-world random streams, object pose/goal randomization, ten
  zero-action settling policy steps, on-palm height retry, and protected-world
  snapshot/restore for masked resets.
- Preserved the reference's `ignore_z_rotation` distinction: online batched task
  rewards ignore Z rotation per world, while the HER callback reproduces the legacy
  batched row-indexing behavior used by Gymnasium Robotics reward recomputation.
- Added task checkpoint/load support for simulation state, goals, filtered actions,
  episode counters, and RNG state.
- Added a native-rigid task benchmark and fixed-state scientific parity tests against
  the CPU Gymnasium Robotics observation ordering and CPU MuJoCo dynamics.
- Added an explicit end-to-end training loop connecting the batched MJWarp task,
  CUDA VecNormalize equivalent, episode-aware future HER replay, and CUDA TQC. Its
  rollout hot path contains no NumPy conversion, per-world loop, SubprocVecEnv, or
  routine device synchronization.
- Matched SB3 rollout storage ordering: terminal observations enter replay before a
  timeout reset, while observation running moments see the post-reset observation.
  Raw transitions remain in replay and are normalized with current statistics when
  sampled.
- Added deterministic fixed-reset-stream evaluation with batched worlds and
  deterministic actor actions. Evaluation reports mean/std reward and final-step
  success rate, then restores training task state.
- Added atomic, versioned trainer checkpoints covering actor/critics/targets,
  optimizers, entropy state, task/simulation state, normalizer, counters, device RNG,
  and optional replay storage.
- Added exact full-replay resume and safe bufferless resume. Bufferless resume resets
  all worlds and enforces at least one fresh complete vectorized episode before HER
  or learner updates can resume.
- Hardened replay planning so every allocation must hold at least one full episode
  per world; auto-capacity now fails with the exact minimum bytes instead of creating
  an unsampleable partial-episode buffer.
- Added `train_gpu.py` with the reference TQC/HER defaults, replay-memory preflight,
  low-frequency CUDA logging, checkpoint/resume, deterministic evaluation, optional
  W&B, compatible study metrics JSON, and `FINAL_SCORE` output.
- Ran real native-rigid direct-MJWarp CLI training and compact-resume smoke jobs. Both
  produced valid final checkpoints and study metrics without invoking the CPU Gym
  environment or stock SB3 rollout/replay infrastructure.
- Added CUDA-event phase profiling around policy inference, simulation/task logic,
  replay insertion, reset/normalization, future-HER sampling, and TQC updates without
  adding synchronization to ordinary training steps.
- Added a complete-loop benchmark using the actual 100-step episode warm-up, the
  `[512, 512, 512]` networks, two critics x 25 quantiles, batch size 2048, and one
  HER/TQC update per batched policy step. Added a matched current-architecture CPU
  benchmark using six MuJoCo `SubprocVecEnv` workers, NumPy HER, and CUDA SB3 TQC.
- Replaced per-substep MJWarp Python launch overhead with a lazily captured Warp CUDA
  graph. The backend owns the Warp stream and uses PyTorch external-stream waits in
  both directions, preserving asynchronous zero-copy interop without host fences.
- Added batch-global contact/collision and per-world constraint high-water monitoring
  plus overflow checks to the complete-loop benchmark. Unsafe measurements cannot be
  selected by auto-tuning.
- Swept complete training at 64, 128, 256, 512, and 1024 worlds on fresh native-rigid
  N=500 and N=1000 models. Both 1024-world runs had zero overflow flags and stayed
  below the measured constraint capacity; 2048 worlds was intentionally not attempted
  because the N=1000 run left only about 755 MiB of device memory.
- Added production `--auto-num-envs` report loading. It requires the exact XML and a
  successful reference-update-ratio complete-loop result, and imports the measured
  contact/constraint capacities together with the fastest world count so the
  selected allocation is reproducible within the benchmarked memory envelope.
- Added direct GPU trainer selection to `generate_and_train.py` while retaining the
  CPU/SB3 default and its existing command behavior. Native-rigid GPU runs emit the
  existing study metrics/`FINAL_SCORE` contract; flex and generated custom-mesh
  `flexcomp` candidates continue to fail closed as explicitly out of scope.
- Added an explicit CPU/GPU trainer field to distributed study specifications and
  queued job payloads. Old study databases acquire the unchanged `cpu` default.
- Extended study manifests with an optional, mutually exclusive `native_task` field
  for built-in `block`, `egg`, or `pen` tasks. Existing `msh_file` manifests retain
  their original validation and CPU behavior.
- Made GPU study initialization fail before enqueue when any job is deformable, any
  manifest row is a custom mesh/flexcomp, no measured world/contact/constraint
  allocation is supplied, or replay memory policy is implicit. CPU host
  `num_envs_per_job` hints are no longer misapplied to direct GPU jobs.
- Updated workers to forward `--trainer gpu`, preserve GPU allocation arguments,
  expand per-job report path placeholders, and launch built-in native tasks without
  treating their names as filesystem paths.
- Kept study scoring and optimizer logic backend-independent. Added the trainer to
  exported per-condition reports and verified a direct GPU metrics payload through
  the unchanged scalar-score calculation.
- Fixed existing NumPy 2.x study scoring compatibility by using `numpy.trapezoid`
  with an older-NumPy fallback; multi-checkpoint CPU or GPU metrics previously failed
  because `numpy.trapz` was removed.
- Audited optimizer/data ratio before the learning validation and found that one GPU
  update per large batched step under-trained relative to the reference's one update
  per six collected transitions. The production default now uses
  `round(num_envs / 6)` updates per GPU vector step; an explicit
  `--gradient-steps` override remains available.
- Made complete-loop timing accumulate all repeated HER/TQC update intervals rather
  than reporting only the final update. Auto-tuning reports must now attest to the
  six-transition reference update ratio, so the earlier one-update-per-batch reports
  cannot be used accidentally for production selection.
- Re-ran N=500 and N=1000 complete-loop tuning at 64, 128, and 256 worlds with the
  reference update ratio. Both models peaked at 128 worlds on this GPU, and both
  recommendations load with the measured 128 contacts/world and 256
  constraints/world allocation.
- Ran a matched-seed N=500 native-block CPU/GPU learning smoke through 12k
  transitions. CPU finished at 12,000 transitions/667 optimizer updates and GPU at
  12,032/693, with matching three-point zero-success curves and final mean reward
  -100. Both produced compatible metrics and resumable checkpoints.
- Added the requested durable validation report at
  `generated/gpu_validation/validation_report.md`, including dependencies, hardware,
  tactile/task parity, corrected performance, learning-smoke results, scope limits,
  exact artifact locations, and the explicit non-claim of statistical equivalence.

## Files changed

- Added `GPU_MIGRATION.md`.
- Added `GPU_MIGRATION_STATUS.md`.
- Added `benchmark_warp_sim.py`.
- Added `requirements-gpu.txt`.
- Added `shadowhand_gpu/__init__.py`.
- Added `shadowhand_gpu/capabilities.py`.
- Added `shadowhand_gpu/model_loader.py`.
- Added `shadowhand_gpu/sensors.py`.
- Added `shadowhand_gpu/warp_backend.py`.
- Added `tests_gpu/__init__.py`.
- Added `tests_gpu/test_capabilities.py`.
- Added `tests_gpu/test_model_loader_and_sensors.py`.
- Added `tests_gpu/test_warp_backend.py`.
- Added `parity_mujoco_warp.py`.
- Added `shadowhand_gpu/parity.py`.
- Added `tests_gpu/test_parity_metrics.py`.
- Added `tests_gpu/fixtures/native_rigid_touch.xml`.
- Added `shadowhand_gpu/rl/__init__.py`.
- Added `shadowhand_gpu/rl/tqc.py`.
- Added `tests_gpu/test_tqc.py`.
- Added `benchmark_tqc.py`.
- Added `shadowhand_gpu/rl/normalization.py`.
- Added `shadowhand_gpu/rl/replay.py`.
- Added `tests_gpu/test_normalization_and_replay.py`.
- Added `benchmark_replay.py`.
- Added `shadowhand_gpu/task.py`.
- Added `tests_gpu/test_task_logic.py`.
- Added `tests_gpu/test_task_gpu.py`.
- Added `benchmark_task.py`.
- Added `shadowhand_gpu/trainer.py`.
- Added `tests_gpu/test_trainer.py`.
- Added `train_gpu.py`.
- Added `benchmark_training.py`.
- Added `benchmark_cpu_training.py`.
- Modified `shadowhand_gpu/warp_backend.py` to execute physics substeps through a
  Warp-owned CUDA graph with explicit Warp/PyTorch stream ordering and a
  `--no-cuda-graphs` diagnostic fallback.
- Modified `shadowhand_gpu/trainer.py` to expose opt-in phase timings and validated
  complete-loop auto-tuning recommendations.
- Modified `train_gpu.py` to support CUDA-graph control and import a benchmarked-safe
  world/contact/constraint allocation.
- Modified `benchmark_task.py` and `benchmark_warp_sim.py` to support CUDA-graph
  comparison runs.
- Modified `benchmark_training.py` to record phase timings, memory, capacity
  high-water marks, and only recommend zero-overflow configurations.
- Modified `generate_and_train.py` to select the CPU or direct GPU trainer.
- Modified `README.md` with the direct GPU workflow, auto-tuning, replay planning,
  resume, and native-rigid scope documentation.
- Added/updated trainer, backend, task, and generator tests for the integrated path.
- Modified `study_common.py` to support explicit native-task manifest rows and
  NumPy-1.x/2.x-compatible metrics integration.
- Modified `optimize_dataset_gpbo.py` to persist/validate the trainer choice, build
  native-task jobs, and expose trainer metadata in reports without changing GPBO.
- Modified `study_worker.py` to route CPU/GPU jobs safely and preserve legacy queued
  mesh payloads.
- Modified `DISTRIBUTED_TRAINING.md` and `README.md` with the native-rigid distributed
  GPU contract and the custom-mesh/flex exclusion.
- Modified `tests/test_study_common.py`, `tests/test_study_worker.py`, and
  `tests/test_optimize_dataset_gpbo.py` for manifest, allocation, command, and metrics
  compatibility coverage.
- Modified `shadowhand_gpu/trainer.py`, `train_gpu.py`, and
  `benchmark_training.py` to preserve and verify the reference optimizer/data ratio
  and sum repeated profiling intervals.
- Modified `benchmark_cpu_training.py` to report measured update counts and
  transitions/update.
- Updated `GPU_MIGRATION.md` from a future design to the implemented native-rigid
  architecture and linked the validation report.
- Added ignored generated artifact `generated/gpu_validation/validation_report.md`
  as required by the migration brief; its durable evidence is duplicated in this
  tracked status file because `generated/` is intentionally gitignored.
- Modified `shadowhand_gpu/rl/normalization.py` to validate and restore normalizer
  checkpoints.
- Modified `shadowhand_gpu/rl/replay.py` to enforce complete-episode capacity and add
  metadata-only clearing for bufferless resume.
- Modified `shadowhand_gpu/task.py` so task checkpoints own cloned state tensors.
- Modified `shadowhand_gpu/warp_backend.py` to expose a derived-state forward pass.
- Modified `shadowhand_gpu/rl/replay.py` to preserve online and legacy HER
  `ignore_z_rotation` reward behavior.
- Modified `shadowhand_gpu/__init__.py` to export the task API.
- Modified `shadowhand_gpu/warp_backend.py` to support matched CUDA state transfer.
- Modified `shadowhand_gpu/warp_backend.py` to enforce native-rigid-only production
  support. Removed the diagnostic `shadowhand_gpu/mjw_compat.py` workaround.
- Updated `GPU_MIGRATION.md` to record that flex/deformable support is out of scope.
- Preserved the pre-existing user modification to
  `train_failed_configs_6_array.sbatch` without editing it.

## Tests run

- `/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m pytest -q`
  - Result: 18 passed in 4.27 seconds.
- `PYTHONDONTWRITEBYTECODE=1 /home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m pytest -q`
  - Expanded result: 22 passed, 1 optional GPU test skipped, 1 NVML warning in
    3.69 seconds.
- `SHADOWHAND_RUN_MJW_TESTS=1 /tmp/shadowhand-mjw.6E7nvV/bin/python -m pytest -q tests_gpu/test_warp_backend.py`
  - Result: 1 passed in 6.01 seconds on the real N=500 model.
- Final post-cleanup reruns:
  - CPU-safe suite: 22 passed, 1 optional GPU test skipped, 1 NVML warning in
    4.86 seconds.
  - Opt-in real N=500 GPU test: 1 passed, 1 NVML warning in 6.74 seconds.
- Phase-C CPU-safe suite: 23 passed, 2 optional GPU tests skipped, 1 NVML warning
  in 4.80 seconds.
- Phase-C opt-in GPU backend/state-transfer tests: 2 passed, 1 NVML warning in
  7.17 seconds.
- Native-rigid scope-gate checkpoint:
  - CPU-safe suite: 23 passed, 3 optional GPU tests skipped, 1 NVML warning in
    5.69 seconds.
  - Opt-in GPU backend suite: 3 passed, 1 NVML warning in 7.99 seconds. This covered
    native-rigid stepping, CUDA state transfer, and fail-closed rejection of the real
    generated N=500 `nflex=1` model.
- TQC focused parity suite: 5 passed in 4.24 seconds.
- TQC checkpoint full CPU-safe suite: 28 passed, 4 optional CUDA/MJW tests skipped,
  1 NVML warning in 8.06 seconds.
- Opt-in CUDA TQC suite: 6 passed, 1 NVML warning in 5.88 seconds. It verifies that
  learner parameters, batches, and returned update metrics remain CUDA-resident.
- Normalization/HER focused CPU/SB3 parity suite: 7 passed in 3.30 seconds.
- Normalization/HER checkpoint full CPU-safe suite: 35 passed, 5 optional CUDA/MJW
  tests skipped in 12.73 seconds.
- Final opt-in CUDA RL suite after the device-resolution fix: 14 passed in 8.91
  seconds. This covers TQC plus normalized future-HER sampling on CUDA.
- Native-rigid task logic/reward focused suite: 13 passed, 1 optional CUDA test
  skipped in 3.44 seconds.
- Native-rigid opt-in CUDA task integration: 1 passed in 13.62 seconds. It covers
  generated native-rigid model loading, reset ranges, CUDA tensor residency,
  CPU/MJWarp 20-substep parity, timeout semantics, protected masked reset, and task
  checkpoint round-trip.
- Phase-G complete CPU-safe suite: 41 passed, 6 optional CUDA/MJWarp tests skipped in
  7.75 seconds.
- Phase-G complete opt-in GPU suite: 27 passed, 2 dependency/reference-only tests
  skipped in 19.58 seconds.
- Trainer orchestration, resume, replay-safety, and metrics focused suite: 18 passed,
  1 optional CUDA test skipped in 6.18 seconds.
- Real native-rigid CUDA task/trainer integration: 1 passed in 24.22 seconds. It
  performs a complete episode, future-HER sample, TQC update, and deterministic
  evaluation without leaving CUDA.
- Phase-H complete CPU-safe suite: 46 passed, 6 optional CUDA/MJWarp tests skipped in
  7.84 seconds.
- Phase-H complete opt-in CUDA/MJWarp suite: 34 passed in 29.73 seconds.
- CUDA-graph backend/task focused verification: 4 passed in 14.78 seconds. This
  included CPU/MJWarp 20-substep state/observation parity, masked reset, and direct
  trainer integration with graph execution active.
- Auto-tuning and generator focused verification: 8 passed in 5.39 seconds; Python
  compilation checks for `train_gpu.py`, `benchmark_training.py`, and the trainer
  also passed.
- Before the update-ratio audit, the loader accepted both diagnostic 1024-world
  reports. They are now deliberately rejected because they used one update per large
  batch; the replacement 128-world reports pass the production gate below.
- Final Phase-H CPU-safe suite: 49 passed, 6 optional CUDA/MJWarp tests skipped in
  10.84 seconds.
- Final Phase-H opt-in CUDA/MJWarp suite: 33 passed, 2 reference/dependency-only tests
  skipped in 14.42 seconds.
- Post-safety-check N=500 complete-loop benchmark smoke: one 64-world result was
  accepted as safe with zero overflow flags, 267 active contacts batch-global, and
  106 maximum constraints/world; the generated report successfully recommended it.
- Phase-I focused study suite: 26 passed in 3.06 seconds. It covers legacy CPU jobs,
  native-task manifests, fail-closed GPU mesh/deformable configurations, explicit GPU
  allocation/replay policy, queue aggregation, worker commands, and GPU metrics
  scoring under NumPy 2.x.
- Native GPU coordinator preview and enqueue smoke: created one rigid-only native
  block job, persisted `trainer=gpu`, and produced a worker command with the expected
  task/candidate/object metadata and 64/128/256 world/contact/constraint allocation.
- Native GPU queued-command equivalent: generated N=10 block MJCF, ran two CUDA
  worlds for four transitions, completed one HER/TQC update, evaluated two episodes,
  saved periodic/final checkpoints and compatible metrics, emitted `FINAL_SCORE`,
  and was ingested by `load_score_from_artifacts` as score 0.0.
- Phase-I full CPU-safe suite: 55 passed, 6 optional CUDA/MJWarp tests skipped in
  7.62 seconds.
- Phase-I full opt-in CUDA/MJWarp suite: 33 passed, 2 reference/dependency-only tests
  skipped in 10.31 seconds.
- Reference-update-ratio helper/auto-report and study focused suite: 27 passed in
  5.75 seconds.
- Both corrected N=500/N=1000 128-world reports were accepted by the production
  recommendation loader as 128 worlds, 128 contacts/world, and 256
  constraints/world. Old reports correctly fail the new update-ratio gate.
- Matched learning smoke completed on both backends and both final checkpoint update
  counts were inspected: CPU 12,000/667 and GPU 12,032/693 transitions/updates.
- Final CPU-safe suite: 57 passed, 6 optional CUDA/MJWarp tests skipped in 5.70
  seconds.
- Final opt-in CUDA/MJWarp suite: 34 passed, 2 reference/dependency-only tests skipped
  in 9.90 seconds.
- Final focused trainer/study routing regression suite: 18 passed in 5.73 seconds;
  `git diff --check` remained clean and the generated validation report existence
  check passed. This includes fail-closed prevention of a queued `trainer_args`
  override bypassing the validated job trainer.
- `train_gpu.py` completed a two-world, four-transition native-rigid smoke run with a
  CUDA TQC update, two deterministic evaluation episodes, atomic final checkpoint,
  metrics JSON, and `FINAL_SCORE`. A second process loaded its compact checkpoint at
  four transitions, collected a fresh complete episode without premature HER, and
  completed at eight transitions.
- `parity_mujoco_warp.py` completed for N=500 no-contact/contact and the temporary
  native rigid-box control model.
- `git diff --check`
  - Result: clean.

## Benchmark results

- Diagnostic-only direct MJWarp N=500 benchmark, RTX 3050 Laptop GPU, 20 physics substeps per
  environment step, cached kernels, 8192 contact/4096 constraint slots per world:
  - 1 world, 5 measured environment steps: 3.716 environment steps/s and
    74.323 physics world-steps/s.
  - 16 worlds, 2 measured batched steps: 17.718 environment steps/s and
    354.364 physics world-steps/s.
  - 64 worlds, 2 measured batched steps: 21.285 environment steps/s and
    425.703 physics world-steps/s.
- These are short simulator-only bring-up measurements from the initial model state;
  they do not yet include randomized resets, policy inference, replay, learning, or a
  matched CPU benchmark.
- Diagnostic-only direct MJWarp N=1000 benchmark with the same protocol (2 measured batched steps):
  - 1 world: 4.005 environment steps/s and 80.101 physics world-steps/s.
  - 16 worlds: 16.471 environment steps/s and 329.428 physics world-steps/s.
  - 64 worlds: 18.331 environment steps/s and 366.625 physics world-steps/s.
- Static replay estimate at one million transitions:
  - N=500: approximately 4.40 GiB.
  - N=1000: approximately 8.13 GiB.
- These totals exceed the local GPU's complete memory before simulator/network
  allocations, so the default CPU replay capacity cannot be used unchanged on this
  GPU.
- CUDA TQC-only benchmark on the RTX 3050 Laptop GPU, batch 2048, two critics × 25
  quantiles, `[512, 512, 512]` actor/critic networks, 2 warmups and 5 measurements:
  - N=500 policy input width 576: 33.398 updates/s, 68,399 samples/s, peak PyTorch
    allocation 226,742,272 bytes (about 216.2 MiB).
  - N=1000 policy input width 1076: 31.016 updates/s, 63,520 samples/s, peak PyTorch
    allocation 250,245,120 bytes (about 238.7 MiB).
  - These measurements exclude simulation, environment logic, and replay storage or
    sampling; they are not end-to-end training throughput.
- CUDA HER replay-only benchmark on the RTX 3050 Laptop GPU, effective capacity
  99,968 across 64 worlds, batch 2048, 2 warmups and 10 measurements:
  - N=500 raw observation width 562: 335.58 batches/s and 687,262 transitions/s;
    persistent allocation 470,879,744 bytes and peak 524,231,168 bytes.
  - N=1000 raw observation width 1062: 289.28 batches/s and 592,454 transitions/s;
    persistent allocation 870,759,936 bytes and peak 968,166,912 bytes.
  - This includes future-goal selection, sparse reward recomputation, current-stat
    normalization, and policy flattening, but excludes TQC and simulation.
- With 50% of the then-free CUDA memory as the explicit safety budget, opt-in
  auto-capacity selected 409,664 transitions for N=500 (4,710 stored bytes each)
  and 221,504 for N=1000 (8,710 bytes each), both across 64 worlds. These values are
  hardware-state snapshots and will be recomputed and recorded per training run.
- Native-rigid task benchmark on the RTX 3050 Laptop GPU, 20 physics substeps per
  environment step, zero-action short rollouts:
  - 1 world: 4.027 environment steps/s, 80.54 physics world-steps/s, and a
    2.627-second all-world randomized/settled reset.
  - 16 worlds: 42.96 environment steps/s, 859.16 physics world-steps/s, and a
    4.089-second all-world reset.
  - 64 worlds: 95.55 environment steps/s, 1,911.07 physics world-steps/s, and a
    6.680-second all-world reset.
- Fresh native-rigid generated high-sensor fixtures at 64 worlds, one warmup and
  three measured task steps, 2,048 contact/constraint slots per world:
  - N=500: 95.90 environment steps/s, 1,918.01 physics world-steps/s, and a
    6.711-second all-world reset.
  - N=1000: 95.50 environment steps/s, 1,909.91 physics world-steps/s, and a
    6.619-second all-world reset.
- These task figures include action conditioning, 20 direct MJWarp steps,
  observation construction, reward, and timeout logic, but exclude replay, policy
  inference, and learner updates. PyTorch-only peak allocation was 330,240 bytes for
  N=500 and 586,240 bytes for N=1000; MJWarp allocations are not included in those
  PyTorch counters.
- The correctness-oriented two-world end-to-end CLI smoke used N=16, horizon two,
  one 4-sample TQC/HER update, and a 16-unit hidden layer. At the second logging
  boundary it reported 0.480 transitions/s including randomized settled reset and
  kernel-warm training startup; deterministic two-episode evaluation took 3.378
  seconds and the measured training/finalization section took 7.031 seconds. This is
  a functional smoke measurement, not a representative production throughput result.
- CUDA graphs changed the fresh N=500 64-world task benchmark from 95.90 to 1096.74
  transitions/s (about 11.4x) while preserving the graph-enabled task/parity tests.
- Current six-worker CPU complete-loop baselines, each after a real complete episode
  and over 20 measured updates:
  - N=500: 56.41 transitions/s, 9.40 updates/s, 1128.17 physics world-steps/s,
    and 1772.79 seconds per 100k transitions.
  - N=1000: 40.36 transitions/s, 6.73 updates/s, 807.30 physics world-steps/s,
    and 2477.40 seconds per 100k transitions.
- Diagnostic collection-heavy graph-enabled configurations with only one optimizer
  update per large vector step peaked at 1024 worlds with 128 contacts/world and 256
  constraints/world. These figures quantify device-resident collection capacity but
  under-train by roughly 171x relative to the six-env reference update/data ratio and
  are not valid matched-learning speedups:
  - N=500: 6556.86 transitions/s, 6.40 updates/s, 131137.11 physics
    world-steps/s, and 15.25 seconds per 100k transitions: 116.24x the CPU
    transition throughput. Mean complete step was 156.19 ms, comprising 114.87 ms
    simulation/task, 4.62 ms HER, and 29.17 ms TQC. Peak PyTorch allocation was
    1,211,394,560 bytes; the 204,800-transition replay used 964,625,184 bytes.
  - N=1000: 4876.87 transitions/s, 4.76 updates/s, 97537.43 physics
    world-steps/s, and 20.50 seconds per 100k transitions: 120.82x the CPU
    transition throughput. Mean complete step was 208.74 ms, comprising 163.74 ms
    simulation/task, 5.67 ms HER, and 32.87 ms TQC. Peak PyTorch allocation was
    2,065,284,096 bytes; the 204,800-transition replay used 1,783,825,184 bytes.
- The diagnostic N=500/N=1000 1024-world capacity high-water marks were respectively
  4289/4291 active contacts batch-global and 117/124 constraints per world, with
  zero overflow flags. The N=1000 backend reported about 791 MB free after
  construction, so 1024 is the largest locally validated collection-heavy batch,
  not the production recommendation or a universal optimum.
- Corrected complete-loop tuning uses `round(worlds/6)` optimizer updates per batched
  step. At 64/128/256 worlds:
  - N=500: 183.26 / 198.48 / 196.25 transitions/s. The 128-world result performs
    32.56 updates/s, 3969.61 physics world-steps/s, and needs 503.83 seconds per
    100k transitions, a matched-update 3.52x speedup over the current CPU pipeline.
    Its mean 650.44 ms step contains 27.89 ms simulation/task, 48.25 ms aggregate
    HER, and 569.03 ms aggregate TQC.
  - N=1000: 152.89 / 161.39 / 159.53 transitions/s. The 128-world result performs
    26.48 updates/s, 3227.74 physics world-steps/s, and needs 619.63 seconds per
    100k transitions, a matched-update 4.00x speedup. Its mean 793.45 ms step
    contains 35.62 ms simulation/task, 83.19 ms aggregate HER, and 670.18 ms
    aggregate TQC.
  - Both selected 128-world results used 21 updates/step, had zero overflow flags,
    and peaked at 111 constraints/world. N=500/N=1000 peak PyTorch allocation was
    361,772,032/493,024,256 bytes.
- The matched-seed 12k-transition learning smoke used 64 GPU worlds and 11
  updates/step so a complete episode entered replay by 6400 transitions and the
  original 8000-transition learning start was preserved. CPU/GPU success curves were
  both `[0, 0, 0]`; reported training-loop times were about 73.0 and 31.746 seconds,
  and observed process wall times were about 88.6 and 37.8 seconds (about 2.34x).
  This validates plumbing and update execution, not learning quality at the full
  16-million-transition budget.
- The Phase-I two-world queued-command smoke reported 1.79 transitions/s at its last
  training log boundary, but it intentionally used a two-step horizon and `[16, 16]`
  network. It is a functional study-contract result, not a performance comparison.

## Known issues and risks

- GPU dependencies remain intentionally absent from the reference Conda environment;
  the validated temporary environment is not a durable user installation.
- Current MuJoCo Warp requires a newer MuJoCo than the CPU environment's 3.3.1 pin;
  use an isolated GPU environment.
- Historical generated custom rigid objects compile as `nflex=1` and remain
  intentionally unsupported. Newly generated `physics_mode=rigid` custom objects use
  an explicit validated/cached mesh-geom conversion and compile with `nflex=0`; they
  are supported with separate provenance. This is not an implicit runtime fallback.
- Rigid custom-mesh dataset studies may now use direct GPU Sobol/GPBO. Deformable
  rows and old cached rigid-flex XML remain CPU-only/fail-closed.
- Large-world training must not retain the conservative CLI defaults of 8192
  contacts and 4096 constraints per world on this 4 GiB device. Use a new
  reference-update-ratio `--auto-num-envs` report or explicitly supply a validated
  allocation. The current production recommendation is 128 worlds with 128/256
  contact/constraint capacity for both tested XMLs; capacities remain workload/XML
  specific and require high-water monitoring during longer experiments.
- Touch vector shape/order is proven. CPU/GPU tactile parity fails for the historical
  rigid-flex object as quantified above; the new rigid-mesh object passes the defined
  one-step and task-step gates but still requires long-run monitoring.
- Existing study XMLs that use `flexcomp` cannot run through the production CUDA
  backend. This is an explicit scope boundary rather than an unguarded fallback.
- Converting the study object to a native rigid mesh is not an acceptable silent
  workaround because MuJoCo collision convexification/contact topology materially
  changes the task and tactile signal.
- No generated N=1000 XML is stored in the repository. The latest validated
  temporary fixture is under `/tmp/shadowhand-native-high.o9xvhI`; tests must
  regenerate it rather than depend on that ephemeral path.
- Candidate alpha/beta labels do not equal realized generator region counts because
  of existing area weighting. Preserve behavior for migration parity.
- `nvidia-smi` reports an NVML driver/library mismatch, although CUDA through PyTorch
  is functional.
- The throughput results are short steady-loop engineering benchmarks, not learning
  validation. No full 16-million-transition CPU/GPU experiment or statistically
  comparable learning curve has been completed yet.
- Replay sampling currently uses `torch.multinomial` over the complete-episode mask.
  It is correct and benchmarked, but remains a profiling candidate at much larger
  capacities if mask scanning becomes material.
- Masked reset restores protected core integration state after globally running the
  reset settling kernels. Contact workspace internals are not checkpointed; the next
  physics step recomputes them from the restored state.
- Reset validity retry performs one low-frequency CUDA-to-host boolean check per
  attempt. Rollout steps, reward calculation, and observation construction remain
  device-resident without scalar synchronization.
- Evaluation currently reuses the training simulation batch to avoid duplicating a
  large MJWarp allocation. Core task/integration state is restored exactly afterward;
  derived contact workspace is recomputed by the subsequent physics step. A separate
  evaluation batch remains an option if profiling or long-run validation shows state
  restoration affects trajectories.
- Full replay checkpoints are deliberately opt-in because they may be hundreds of
  MiB or more. Compact checkpoints restore all learning/normalization state, start
  fresh episodes, and enforce a complete-episode warm-up, but cannot reproduce the
  exact next update as a full checkpoint can.

## Exact next task

Run the remaining scientific experiment rather than another implementation phase:
execute multiple matched seeds for native-rigid N=500 through a substantial fraction
of the 16-million-transition budget, preserve the reference update/data ratio, and
compare success distributions/AULC and candidate ranking stability while monitoring
long-run contact/constraint high-water marks. Repeat N=1000 after N=500 is stable.
The current 12k smoke is only a pipeline gate. Flex/custom-mesh GPU support remains
out of scope, so existing mesh studies continue on the CPU reference.

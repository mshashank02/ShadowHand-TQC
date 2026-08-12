# ShadowHand Direct MuJoCo Warp Migration

## Purpose

This document is the correctness contract for migrating the existing CPU MuJoCo +
SB3-Contrib TQC/HER training pipeline to direct MuJoCo Warp simulation and
CUDA-resident reinforcement learning. `ShadowHand_TQC.py` remains the CPU reference
implementation and scientific comparison baseline. The native-rigid parity gates
now pass. Phase II adds validated custom rigid mesh geoms; compiled rigid-flex and
deformable-flex models remain explicitly unsupported by the production GPU backend.

The migration is intentionally staged. A faster result is not accepted if it changes
the generated model, task distribution, tactile observation, action semantics, reward,
HER behavior, TQC objective, or study output contract.

## Current pipeline

1. `study_common.py` maps a candidate `(N, alpha, beta)` to ratios passed into the
   sensor generator.
2. `pipeline_generate.py` and the `Generate*` modules write touch sites, touch
   sensors, and a complete MuJoCo XML.
3. `generate_and_train.py` launches `ShadowHand_TQC.py` for the generated XML.
4. `DynamicXMLTouchEnv` wraps Gymnasium Robotics' CPU MuJoCo manipulation task.
5. `TimeLimit`, `Monitor`, `TimeFeatureWrapper`, and `VecNormalize` construct the
   observation/reward stream.
6. SB3-Contrib TQC trains from an SB3 `HerReplayBuffer`; simulation and replay are
   CPU-side and sampled tensors are copied to CUDA.
7. Metrics JSON and the `FINAL_SCORE` line are consumed by the study worker.

## Scientific invariants

The GPU path must preserve all of the following unless a separately reviewed change
explicitly updates the experiment definition:

- The generated XML, mesh/flex object description, touch site positions, compiled
  touch sensor order, and raw scalar `sensordata` values.
- Twenty MuJoCo substeps per policy step, model timestep `0.002`, and the resulting
  policy timestep `0.04` seconds.
- The current 20-dimensional actuator mapping, optional clipping, action scaling,
  exponential smoothing, and default absolute-control behavior.
- Object and goal randomization, including quaternion convention, reset settling,
  and per-environment seed independence.
- Seven-dimensional achieved and desired goals, sparse reward, position threshold
  `0.01`, rotation threshold `0.1`, and timeout semantics.
- The 100-step horizon, the time feature, observation dictionary keys/order, and
  `VecNormalize` running-statistic behavior.
- TQC defaults used by the project: two critics, 25 quantiles per critic, two top
  quantiles dropped per critic, batch size 2048, gamma 0.95, tau 0.05, learning rate
  1e-3, automatic entropy, target entropy -20, train frequency 1, and one gradient
  step per six transitions collected by the reference six-environment vector step.
  The GPU default uses `round(num_envs / 6)` sequential updates per batched policy
  step to preserve that update/data ratio.
- HER defaults: four sampled goals, 80 percent virtual transitions, completed
  episodes only, inclusive `future` goal selection, goal values from future
  `next_observations["achieved_goal"]`, reward recomputation from the sampled
  transition's next achieved goal, and timeout dones masked for bootstrapping.
- Metrics JSON fields and `FINAL_SCORE` output used by the study tooling.

### Existing sensor-allocation discrepancy

Candidate labels imply target palm/finger/tip counts, but the generator converts the
ratios using surface-area weights (`Ap=6557`, `Apx=26885`, `At=7193`). Consequently,
realized counts do not equal the nominal alpha/beta fractions. For example, the
existing N=500, alpha=0.3, beta=0.6 candidate realizes 78/301/121 rather than the
nominal 150/140/210 grouping.

This behavior predates the GPU migration. The parity path must preserve the realized
generated XML and must not silently fix the allocation formula. Any correction is a
separate scientific change requiring regenerated datasets and explicit provenance.

## Compatibility findings

### MuJoCo Warp

The target is the direct MuJoCo Warp API:

```python
model = mujoco.MjModel.from_xml_path(xml_path)
warp_model = mujoco_warp.put_model(model)
warp_data = mujoco_warp.make_data(model, nworld=batch_size)
mujoco_warp.step(warp_model, warp_data)
```

Historically, the custom rigid object was emitted as `flexcomp type="gmsh" dim="3"`
with `rigid="true"`; compiled models report `nflex=1`. It is an ordinary free rigid
body with no elastic DOFs, but collision is performed through the rigid-flex shell.
Controlled diagnostics show that MJWarp creates 2,674 flex-flex self contacts after
its first collision pass even in a CPU no-contact pose, despite `selfcollide="none"`.
The old path therefore fails the contact/tactile gate and remains diagnostic-only.

For `physics_mode=rigid`, Phase II deterministically extracts the exterior triangles
of the source tetrahedra and emits the same free body with a conventional
`geom type="mesh"`. This compiled `nflex=0` model is supported. For
`physics_mode=deformable`, the existing flex generation is unchanged and remains
excluded from production CUDA. Capability is decided from compiled structure, not
filenames or command-line labels. The installed CPU environment uses MuJoCo 3.3.1
while current MuJoCo Warp requires a newer MuJoCo release, so GPU dependencies must
remain isolated from the reference Conda environment.

Current models have 36 velocity DOFs, below MuJoCo Warp's documented warning region
for models above roughly 60 DOFs. XML `nconmax`/`njmax` are not relied upon; GPU code
configures batch allocation limits explicitly and reports high-water/overflow state
in benchmarks.

### Touch sensors

MuJoCo Warp exposes batched `Data.sensordata` and contact-sensor matching. The GPU
path builds a compiled sensor layout from sensor names, types, `sensor_adr`, and
dimensions. It must not assume a sensor ID is a scalar `sensordata` address, even
though all sensors in the current N=500 model are scalar and that assumption happens
to hold there.

The actual N=500 model has 529 scalar sensor values: 29 non-touch values followed by
500 touch values. The CPU observation before `TimeFeatureWrapper` has `N + 61`
values; after the wrapper it has `N + 62`. The policy also receives both seven-value
goal keys, giving flattened policy inputs of 576 for N=500 and 1076 for N=1000.

## Production GPU data flow

```text
generated XML
    -> one MuJoCo model + batched MuJoCo Warp Data on CUDA
    -> cached zero-copy PyTorch views of qpos/qvel/ctrl/sensordata
    -> CUDA action transform, stepping, goals/reward/termination/reset
    -> CUDA observation construction and running normalization
    -> CUDA episode-aware replay and vectorized HER sampling
    -> CUDA TQC actor/critic/entropy updates
    -> CPU only for periodic logs, checkpoints, evaluation artifacts, metrics JSON
```

Warp arrays are exposed through cached `wp.to_torch` views. A Warp-owned CUDA stream
and PyTorch external stream exchange device-side waits without host fences. One
MJWarp step is captured lazily and replayed for all 20 physics substeps. The ordinary
hot path contains no `.cpu()`, `.numpy()`, per-world Python loop, or routine device
synchronization.

## Batched reset design

The environment owns device-side per-world done masks, RNG state, goals, and
filtered-action state. The current timeout-only task keeps worlds
horizon-synchronized; masked reset is still implemented and protects unselected
integration state while global settling kernels run. Reset reproduces the randomized
initial pose, randomized target, action-state clearing, ten zero-action settling
policy steps, and valid-height retry behavior.

Random draws must be reproducible for a fixed global seed and world index while
remaining statistically independent across worlds. CPU/GPU samples need not be
bitwise identical, but distributions and deterministic replay within each backend
must be tested.

## TQC migration

The PyTorch implementation is decomposed into actor, quantile critics, loss, entropy
coefficient, and update orchestration and is parity-tested against SB3-Contrib 2.7.1:

- squashed diagonal Gaussian actor with log standard deviation clamped to [-20, 2];
- two critics with 25 quantiles each;
- flatten and sort 50 target quantiles, dropping the largest four;
- entropy-adjusted Bellman target and pairwise quantile Huber regression;
- actor objective `mean(alpha * log_prob - mean_Q)`;
- learned log entropy coefficient initialized to zero with target entropy -20; and
- Polyak target updates every gradient step with tau 0.05.

Fixed-weight forward, loss, gradient, and one-step optimizer parity tests compare the
new modules with the installed SB3-Contrib source.

## HER and replay migration

A purpose-built CUDA ring buffer is used because it expresses the exact SB3
future-HER semantics without CPU callbacks or opaque framework adaptation. It tracks
episode starts/lengths, invalidates an overwritten episode before reuse, samples
complete episodes only, and performs vectorized future-goal replacement and reward
recomputation.

Replay capacity is a hardware constraint, not just a hyperparameter. Conservative
full-transition estimates are:

| Sensor count | Bytes/transition | 1M transitions |
| ---: | ---: | ---: |
| 500 | about 4,728 | about 4.40 GiB |
| 1000 | about 8,728 | about 8.13 GiB |

These totals exclude MuJoCo Warp state, networks, activations, and optimizer state.
The local RTX 3050 exposes about 3.68 GiB, so a 1M CUDA replay cannot fit. The GPU
trainer estimates allocation before construction, fails with a useful diagnostic
when an explicit capacity cannot fit, and offers an opt-in auto-capacity mode that
records the chosen capacity. It does not silently alter a requested value.

## Parity gates

1. Load and step supported rigid-geom XML (primitive and mesh) in CPU MuJoCo and
   direct MuJoCo Warp; verify that compiled flex XML fails closed before allocation.
2. Export and compare compiled sensor maps and tactile vector order.
3. Starting from matched state and control, compare qpos, qvel, sensordata, contact
   counts, and task observations over one and multiple policy steps.
4. Validate action transforms, goal distances, sparse rewards, success, timeout, and
   masked resets.
5. Validate time-feature and normalization updates/outputs against SB3.
6. Validate replay episode bookkeeping and HER sample/reward semantics.
7. Validate TQC forward/loss/update results under fixed parameters and random draws.
8. Run short fixed-seed CPU and GPU training smoke tests and compare learning-facing
   metrics within documented stochastic tolerances.

MuJoCo Warp uses float32 and GPU atomics, so bitwise CPU agreement is not expected.
Tolerances will be selected from measured error and recorded with each parity test;
they will not be widened merely to hide divergent dynamics.

## Phase II rigid custom-object contract

The converter reads the study's GMSH 2.x ASCII or binary tetrahedral volume mesh,
counts each tetrahedron face by its unordered vertex IDs, removes twice-occurring
internal faces, and winds each once-occurring face away from its opposite tetrahedron
vertex. It does not recenter, rotate, mirror, smooth, simplify, or convexify the
geometry. Output validation checks source/output bounds, volume and centroid,
surface area, connected components, watertightness, manifold incidence, and winding.
Conversion fails closed and is cached by source SHA-256, converter version, and
conversion parameters. The tactile candidate is intentionally absent from this key.

The rigid MJCF retains `object`, `object:joint`, the free joint, body position,
explicit mass, center of mass, and diagonal inertia. It maps the compiled old-flex
defaults to the mesh geom: friction `1 0.005 0.0001`, `condim=3`, `solref=0.02 1`,
`solimp=0.9 0.95 0.001 0.5 2`, margin/gap zero, contact masks `1/1`, and priority
zero. Flex radius, flex self-collision, and flex internal-contact flags have no exact
rigid-geom analogue and are recorded in `rigid_object_representation.json` rather
than silently approximated.

The backend execution gate for a matched one-step settled-contact state is qpos
maximum error at most `1e-5`, qvel at most `1e-3`, tactile maximum error at most
`0.05`, total tactile magnitude error at most 2%, active-sensor Jaccard at least
0.9, and exact contact count/geom pairs. The 20-substep task gate allows qpos
`3e-4`, qvel `0.02`, and touch `0.2`, while requiring exact sparse reward and
success. These bounds are grounded in the measured sensor magnitude and activation
pattern; they are execution gates, not a claim of long-horizon trajectory identity.

The representative N=500 mesh passes: settled one-step qpos error is `6.26e-7`,
qvel `4.45e-4`, touch maximum `0.03470`, touch RMSE `0.00155`, total tactile
magnitude error `0.876%`, active Jaccard `1.0`, and all three geom pairs match.
This is a dramatic improvement from the old rigid-flex touch error `74.35`, but it
is looser than the native-block control (`1.48e-5` touch). The conclusion is
Outcome B: rigid-flex collision was the dominant discrepancy, while conventional
mesh contact remains a measurable backend difference requiring long-run monitoring.

Changing representations is also a scientific experiment change. On CPU, the old
settled rigid flex has 89 contacts versus three for the new mesh and produces a very
different tactile field/trajectory, even though mass and inertia are exact. New
rigid-mesh runs therefore need distinct provenance and cannot be pooled with
historical rigid-flex CPU results.

## Benchmark protocol

Raw simulation benchmarks use the real model, warm up all kernels, synchronize only
at measurement boundaries, and report policy steps/s, physics world-steps/s, batch
size, substeps, contact allocation, GPU, dependency versions, and peak memory. Safe
batch sizes begin at 1, 16, and 64 and increase only while memory headroom remains.

Integrated benchmarks then measure environment plus replay sampling and TQC updates,
including transfers and synchronization. The comparison baseline is the existing CPU
pipeline at its current six-environment default. OOM limits and failed compatibility
gates are results and must be reported rather than hidden.

## Repository layout

```text
shadowhand_gpu/
  capabilities.py
  model_loader.py
  parity.py
  rigid_flex_diagnostic.py
  sensors.py
  warp_backend.py
  task.py
  rl/normalization.py
  rl/replay.py
  rl/tqc.py
train_gpu.py
benchmark_warp_sim.py
diagnose_rigid_collision.py
diagnose_rigid_flex.py
object_conversion/
  gmsh_to_rigid_surface.py
requirements-gpu.txt
tests_gpu/
```

`generate_and_train.py`, the study coordinator, and study workers expose an explicit
CPU/GPU trainer selector. GPU studies are rigid-only and accept either `native_task`
or validated `msh_file` rows plus a measured allocation. Deformable flex studies
stay on CPU.

## Milestones

1. Architecture/status checkpoint and green CPU baseline.
2. Optional-dependency capability report, compiled sensor map, real-model MJWarp
   load/step test, and raw simulation benchmark.
3. Batched actions, observations, goals, rewards, terminations, and independent reset.
4. Tactile and CPU/GPU environment parity for supported rigid-geom fixtures.
5. CUDA normalization and episode-aware HER replay with memory planning.
6. Exact PyTorch TQC and SB3 parity tests.
7. End-to-end trainer, checkpoint/resume, evaluation, metrics compatibility, and
   study backend integration.
8. Performance profiling, CUDA graphs, reference-update-ratio auto-tuning, and
   complete documentation. Flex/deformable support is out of scope.
9. Deterministic tetrahedral exterior conversion, rigid mesh MJCF, controlled old-
   flex diagnostics, custom-object parity, learning smokes, and real-object tuning.

Milestones 1-9 are complete for supported rigid-geom models. See
`GPU_MIGRATION_STATUS.md` for commands and checkpoint-by-checkpoint evidence, and
`generated/gpu_validation/validation_report.md` for the fixed-seed N=500 report.
Long multi-seed learning/ranking validation remains future experiment execution, not
an unimplemented training component.

## References

- [Official MuJoCo Warp repository](https://github.com/google-deepmind/mujoco_warp)
- [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [Warp/PyTorch interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)

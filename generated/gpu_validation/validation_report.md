# ShadowHand Rigid-Geom GPU Validation

Date: 2026-08-11

## Phase IV: 2D OBJ rigid-flex CPU gate

On 2026-08-13 the exact audited exterior of the worst
`obj_size-large_ar-high_macro-high_rough-high` object was loaded as
`flexcomp type="mesh" dim="2" rigid="true"`, retaining the original 0.03125
scale, 1.25 mm physical flex radius, object body/free joint, explicit
mass/COM/inertia/damping, and compiled contact parameters. This is a non-convex
triangular flex, not an ordinary convexified mesh geom. It compiles in CPU MuJoCo
3.3.1 to 1666 vertices, 4992 edges, and 3328 triangles, all attached to one object
body with six global DOFs.

The source audit confirms that this representation avoids both previously
localized tetrahedral Warp mechanisms: `_flex_tet_internal_collisions_detect`
returns for `dim != 3`, and the four-faces-per-tetrahedron narrowphase is separate
from the `dim=2` `_flex_narrowphase_unified` triangle path.

The CPU representation gate is **Outcome A**. Contact-free motion agrees exactly
at 1, 10, and 100 steps. Across the unchanged fingertip, palm, deepest-concavity,
macro, and roughness fixtures, maximum onset error is 0.0000016764 mm, all
contact/constraint-row counts match, maximum contact-position error is 0.018292 mm,
maximum normal error is below 0.00769 degrees, and active tactile Jaccard is 1.0.
Common-pose total tactile relative errors are respectively 1.708%, 7.938%, 10.239%,
9.577%, and 9.971%, within the predeclared 25% representation gate. Full model,
onset, manifold, force, and tactile artifacts are in
`generated/rigid_flex_2d_validation/`.

This CPU pass authorizes controlled CPU-versus-Warp 2D-flex validation only. It
does not yet authorize N=1000, performance claims, production integration, or RL.

## Scope and conclusion

The original Phase-I portion of this report validates the direct MuJoCo Warp + CUDA
PyTorch path on the N=500,
`alpha=0.3`, `beta=0.6` tactile layout (`Rppx=1.071429`, `Rpt=0.714286`) with the
built-in native-rigid block. The CPU and GPU learning smokes used the exact same
generated XML, seed, goal/reward semantics, episode horizon, batch size, network,
and TQC/HER hyperparameters.

The implementation passes its component and short-rollout parity gates and improves
reference-update-ratio complete-loop throughput by 3.52x for N=500 and 4.00x for
N=1000 on this laptop. A 12,000-transition matched-seed smoke crossed
`learning_starts` and produced nearly matched optimizer counts, compatible metrics,
and checkpoints on both backends. Both success curves remained zero. This run is far
too short relative to the 16-million-transition experiment to establish comparable
learning distributions or candidate rankings, so scientific learning equivalence is
not claimed.

Phase II, documented below without removing the Phase-I results, replaces the custom
rigid object's collision representation with a validated conventional mesh geom.
Rigid custom `.msh` objects now pass the defined GPU execution gate. Compiled
rigid-flex and deformable-flex models remain excluded.

## Hardware and software

| Component | CPU reference | Direct GPU path |
|---|---|---|
| CPU | 12th Gen Intel Core i7-12650H, 10 cores / 16 threads | Same host |
| GPU | RTX 3050 Laptop GPU for SB3 TQC | RTX 3050 Laptop GPU, 3.68 GiB usable |
| MuJoCo | 3.3.1 | 3.11.0 |
| MuJoCo Warp | Not used | 3.11.0 |
| NVIDIA Warp | Not used | 1.16.0 |
| PyTorch / CUDA runtime | 2.4.0+cu121 / 12.1 | 2.4.0+cu121 / 12.1 |
| SB3 / SB3-Contrib | 2.7.1 / 2.7.1 | Behavioral reference only |
| NumPy | 2.4.6 | No transition hot-path dependency |

`nvidia-smi` reported an NVML driver/library mismatch during this work, but PyTorch
and Warp both executed successfully on CUDA.

## Candidate and observation contract

| Field | Value |
|---|---:|
| Layout | N=500, alpha=0.3, beta=0.6 |
| Generated ratio arguments | Rppx=1.071429, Rpt=0.714286 |
| Object | Built-in native-rigid block |
| MuJoCo state | nq=38, nv=36, nu=20 |
| Scalar sensor data | 529 |
| Touch values | 500, deterministic `sensordata[29:529]` |
| Raw observation width | 562 |
| Policy input width | 576 |
| Goal fields | achieved_goal=7, desired_goal=7 |

The production path reads the real MuJoCo touch sensors through cached zero-copy
Warp/PyTorch CUDA views. It does not use binary contact flags or collapse sensor
layouts.

## Physics and task parity

- N=500 free/no-contact one-step CPU MuJoCo versus MJWarp: qpos maximum absolute
  error `1.61e-8`, qvel `5.12e-7`, tactile error zero.
- Native-rigid settled-contact control fixture: qpos maximum error `5.07e-8`, qvel
  `3.12e-6`, touch `1.48e-5`.
- The opt-in task test also passes matched CPU/MJWarp state and observation checks
  over one 20-substep policy action, action/goal/reward/success/timeout semantics,
  masked reset protection, and checkpoint round-trip.
- The historical generated custom rigid-flex study object does not pass contact parity (recorded
  diagnostic errors: qpos `0.00161`, qvel `0.804`, touch `74.35`) and is rejected by
  the production backend.

CUDA-graph execution raises the N=500 64-world task-only result from 95.90 to
1096.74 transitions/s (21,934.84 physics world-steps/s). These task-only numbers
exclude replay and learning and are not used as the primary speedup claim.

## Complete-loop throughput with reference update ratio

The current CPU default collects six transitions and performs one SB3 TQC update per
vector step. GPU measurements therefore use `round(num_envs / 6)` updates per
batched step. This ratio is now the production default and a required property of
auto-tuning reports. Earlier one-update-per-large-batch measurements are retained in
`GPU_MIGRATION_STATUS.md` as collection diagnostics only.

All measurements use a real completed 100-step episode before timing, batch 2048,
future HER (`n_sampled_goal=4`), `[512,512,512]` networks, two critics x 25
quantiles, 20 physics substeps/action, CUDA graphs, 128 contacts/world, and 256
constraints/world.

| Sensors | Backend/config | Transitions/s | Updates/s | Physics world-steps/s | Seconds/100k | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 500 | CPU MuJoCo, 6 envs | 56.41 | 9.40 | 1,128.17 | 1,772.79 | 1.00x |
| 500 | MJWarp, 128 worlds, 21 updates/step | 198.48 | 32.56 | 3,969.61 | 503.83 | 3.52x |
| 1000 | CPU MuJoCo, 6 envs | 40.36 | 6.73 | 807.30 | 2,477.40 | 1.00x |
| 1000 | MJWarp, 128 worlds, 21 updates/step | 161.39 | 26.48 | 3,227.74 | 619.63 | 4.00x |

The tested N=500 throughput was 183.26, 198.48, and 196.25 transitions/s at 64,
128, and 256 worlds. N=1000 measured 152.89, 161.39, and 159.53 transitions/s.
Thus 128 worlds is the fastest measured reference-ratio configuration for both
sensor counts on this GPU.

At N=500/128, the mean 650.44 ms complete step comprised 27.89 ms simulation/task,
48.25 ms HER, and 569.03 ms TQC. At N=1000/128, 793.45 ms comprised 35.62 ms
simulation/task, 83.19 ms HER, and 670.18 ms TQC. Learning is now the measured
bottleneck rather than simulator launch overhead.

Both 128-world runs had zero overflow flags. N=500/N=1000 respectively reached
535/534 batch-global active contacts and 111/111 maximum constraints per world.
Peak PyTorch allocations were 361,772,032 and 493,024,256 bytes; their short
benchmark replay allocations were 120,578,848 and 222,978,848 bytes.

## Fixed-seed learning smoke

Common settings: N=500 native block XML, seed 0, horizon 100, 12,000 requested
transitions, learning start 8,000, replay request 20,000, batch 2048, gamma 0.95,
learning rate 1e-3, tau 0.05, automatic entropy, future HER with four sampled goals,
`[512,512,512]`, two critics, and deterministic 10-episode evaluations around 1/3,
2/3, and completion.

| Result | CPU reference | Direct GPU |
|---|---:|---:|
| Parallel environments/worlds | 6 | 64 |
| Gradient updates/vector step | 1 | 11 |
| Final transitions | 12,000 | 12,032 |
| Final optimizer updates | 667 | 693 |
| Transitions/update | 17.99 over whole run; 6 after learning starts | 17.36 over whole run; 5.82 after learning starts |
| Evaluation checkpoints | 0.3335, 0.6670, 1.0 | 0.3360, 0.6720, 1.0 |
| Success curve | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Final mean reward | -100.0 | -100.0 |
| Final success | 0.0 | 0.0 |
| Reported training-loop time | about 73 s | 31.746 s |
| Observed process wall time | about 88.6 s | about 37.8 s |
| Final checkpoint | SB3 model + VecNormalize | CUDA trainer state |

The short-run process speedup was approximately 2.34x including startup,
evaluations, and serialization. The purpose of this run was to verify that learning
actually starts and the metric/checkpoint contracts survive a nontrivial update
count. A zero sparse-reward curve after only 12k/16M transitions is expected to be
uninformative; it neither proves nor disproves learning equivalence.

Temporary artifacts from this validation session:

- CPU: `/tmp/shadowhand-validation-n500/cpu`
- GPU: `/tmp/shadowhand-validation-n500/gpu`
- Reference-ratio benchmark reports:
  `/tmp/shadowhand-training-benchmark-n{500,1000}-w{64,128,256}-reference-updates.json`

These `/tmp` paths are ephemeral. The durable commands and numeric results are also
recorded in `GPU_MIGRATION_STATUS.md`.

## Custom rigid collision representation study (Phase II)

### Model and source characterization

The real `n0500_a0p3_b0p6` large/high/high/high object historically used one free
joint plus `flexcomp type="gmsh" dim="3" dof="trilinear" rigid="true"`. It has no
elasticity and adds no object DOFs: the object contributes only 7 qpos/6 qvel from
the free joint. The compiled flex has 2,387 vertices, 12,602 edges, 8,552 tetrahedra,
and a 3,328-triangle collision shell; all vertices reference object body 28.

The source is little-endian binary GMSH 2.2 with 2,387 nodes and 8,552 first-order
tetrahedra and no explicit surface elements. Deterministic exterior extraction gives
1,666 compacted boundary vertices and 3,328 triangles. The output is one watertight
component with zero boundary, non-manifold, or winding-mismatch edges. Raw bounds
are preserved exactly; relative volume error is `2.16e-15` and volume-centroid error
is `3.53e-16`. At the existing scale, dimensions remain
`0.06791155 x 0.06152016 x 0.12079550` metres.

All 24 `sphere_study_v1` sources passed the same conversion checks. They produce 24
distinct source/cache hashes and retain differences across small/medium/large size,
low/high aspect ratio, macro geometry, and roughness. Converted surfaces span
1,594-2,418 vertices and 3,184-4,832 triangles.

The new generated object has `nflex=0`, one free joint, one mesh geom, and the same
body pose, mass `0.976562`, center of mass, and diagonal inertia
`0.00305176 0.00305176 0.00305176`. Friction, contact dimension, solver parameters,
margin/gap, contact masks, and priority are mapped explicitly. Flex radius and flex
self/internal-collision flags have no direct mesh-geom counterpart and are recorded
as unmapped semantics in the representation manifest.

### Controlled old rigid-flex failure localization

The diagnostic-only old-path harness works around MJWarp 3.11's separate `nJfe=0`
edge-Jacobian illegal write only after proving every flex vertex belongs to the same
rigid body. The production backend is unchanged and continues to reject flex.

Five fixtures cover free motion, one approaching geom, isolated fingertip contact,
isolated palm contact, and the settled default state. Imported initial contacts match
CPU to roughly `3e-8` metres. After Warp's first own collision pass, however, it
creates 2,674 flex-flex contacts even with the object at z=2 and no CPU contacts.
Those contacts report `geom=(-1,-1)` and `flex=(0,0)`, despite
`selfcollide="none"`. In the approach fixture Warp diverges at step 1; CPU's intended
fingertip contact first occurs at step 5. The settled comparison becomes 89 CPU
contacts versus 2,757 Warp contacts in one step. This localizes the original failure
to collision identification before tactile accumulation or learning.

### Three-way backend comparison

All values below are maximum absolute errors for one matched CPU/MJWarp step after
200 CPU settling steps, except the no-contact row noted in the text.

| Representation | Backend comparison | qpos error | qvel error | touch error | Notes |
|---|---|---:|---:|---:|---|
| Native block geom | CPU vs Warp | `5.07e-8` | `3.12e-6` | `1.48e-5` | Phase-I control |
| Old rigid flex | CPU vs Warp | `0.001607` | `0.8036` | `74.35` | 89 vs 2,757 contacts after Warp collision |
| New rigid mesh geom | CPU vs Warp | `6.26e-7` | `4.45e-4` | `0.03470` | 3/3 contacts and geom pairs match |

New-mesh free motion remains tight: qpos `1.61e-8`, qvel `5.12e-7`, and touch zero.
In the settled-contact test, tactile mean/median/RMSE are
`6.94e-5 / 0 / 0.00155`; both backends activate the same one sensor (Jaccard 1.0),
Pearson correlation is approximately 1.0, and total magnitude is 3.9616 CPU versus
3.9963 Warp (0.876% relative error). The largest error is 0.03470 at
`robot0:TS_ffproximal_auto_006`. Contact position, normal, distance, and force
maximum errors are respectively `1.33e-4 m`, `0.00291`, `1.95e-7 m`, and `0.05365`.

The acceptance gate requires settled qpos <=`1e-5`, qvel <=`1e-3`, touch maximum
<=`0.05`, total touch error <=2%, active Jaccard >=0.9, and exact contact count/geom
pairs. A full 20-substep policy action also passed qpos <=`3e-4`, qvel <=`0.02`,
touch <=`0.2`, and exact achieved-goal reward/success checks. These are one-step and
task-step execution gates, not proof of long-horizon trajectory identity.

The outcome is **B**. Conventional mesh collision removes the dominant rigid-flex
failure and makes tactile error less than 1% of the active CPU signal in this gate,
but it is not as tight as the native-block control. Longer rollouts and multi-seed
candidate-ranking stability remain open scientific validation.

### CPU old representation versus CPU new representation

Mass, center of mass, and inertia match exactly, but collision semantics do not. In
independently settled CPU trajectories the old/new representations have 89/3
contacts, qpos differs by about 0.883, qvel by 6.21, and total tactile magnitude is
about 501.65 versus 4.06. From the exact old settled state, a single step differs by
1.04 in qvel and the old/new tactile totals are 435.19/0. This is not scientifically
interchangeable with historical rigid-flex data. New runs require distinct
`rigid_mesh_geom` provenance and must not be pooled with old results.

### Real-object learning and throughput

The N=500 real-object smoke used 64 worlds, 12,000 requested transitions, the
unchanged `[512,512,512]` TQC/HER learner, batch 2,048, learning start 8,000, and 11
updates per vector step. It finished at 12,032 transitions with 64 completed
episodes, 693 optimizer updates, three evaluations, metrics JSON, two periodic
checkpoints, and a final checkpoint in 37.48 seconds. Its zero success curve is
execution evidence only, not learning-equivalence evidence.

The matched-update complete-loop N=500 sweep on the real mesh measured:

| Worlds | Transitions/s | Updates/s | Physics world-steps/s | Seconds/100k |
|---:|---:|---:|---:|---:|
| 32 | 181.21 | 28.31 | 3,624.25 | 551.84 |
| 64 | 180.57 | 31.04 | 3,611.46 | 553.79 |
| 128 | **195.21** | **32.03** | **3,904.17** | **512.27** |
| 256 | 192.11 | 32.27 | 3,842.26 | 520.53 |

The same new XML on the current six-env CPU loop measured 75.24 transitions/s, so
the 128-world GPU result is 2.59x faster. Its mean step comprises 37.47 ms
simulation/task, 48.67 ms HER, and 565.16 ms TQC; TQC remains dominant. Peak
PyTorch allocation is 360,539,136 bytes, contact/collision high-water is 513/2,171,
maximum constraints are 75 per world, and overflow flags are zero.

N=1000 compiles with exactly 1,000 touch channels, raw observation width 1,062,
policy input width 1,076, and a 19,968-transition/173.9 MB replay in the learning
smoke. That smoke also completed 12,032 transitions, 64 episodes, 693 updates,
evaluations, metrics, and checkpoints. At 128 worlds the complete loop measures
160.24 transitions/s, 26.29 updates/s, 3,204.73 physics world-steps/s, and 624.08
seconds/100k; contact/collision high-water is 518/2,170, maximum constraints are 75,
and overflow flags are zero.

The durable machine-readable Phase-II summary is
`generated/gpu_validation/phase2_rigid_mesh_summary.json`.

## Remaining scientific validation

- Run multiple seeds through a substantial fraction of the 16-million-transition
  budget and compare success distributions/AULC, not individual trajectories.
- Monitor contact/constraint high-water marks over those long randomized runs.
- Compare candidate ranking stability across multiple rigid mesh objects and tactile layouts.
- Do not apply rigid-mesh conclusions to deformable or historical rigid-flex runs.

## Convexity / Collision Geometry Audit

The deterministic 100,000-area-sample audit of exactly the 24
`sphere_study_v1` rigid surfaces classifies the current single-convex-hull
collision representation as **Outcome C (material)**:

MuJoCo 3.11.0 directly compiled the representative 1,666-vertex/3,328-face OBJ
with a 989-vertex/1,974-face convex graph at `mesh_graphadr=0`, exactly matching
the audit hull. The CPU reference environment remains MuJoCo 3.3.1.

- maximum volume inflation: **4.423%**;
- maximum p95 hull gap: **2.551 mm**;
- maximum p99 hull gap: **3.420 mm**;
- absolute sampled maximum gap: **4.459 mm** (4.460 mm at 250k convergence;
  deterministic vertex maximum 4.459 mm);
- maximum affected surface above 0.1 mm: **41.47%**;
- worst object: `obj_size-large_ar-high_macro-high_rough-high`;
- macro distinction preserved: **yes, but incompletely** — matched hulls retain
  3.993-8.367 mm p95 separation, while macro-high concavities are systematically
  bridged;
- roughness geometrically affected: **yes** — roughness selects distinct meshes,
  not friction, and convexification removes inward roughness/concavity;
- full 24-object multi-seed learning validation: **not approved as the next phase**
  until a targeted collision-representation follow-up preserves the study factors.

The previous manual large/high/high/high hull view was not quantitatively reliable:
that object is the measured worst case. See the full method, all-object table,
convergence study, factor analysis, hashes, and four-view heatmaps in
`generated/convexity_audit/convexity_audit.md`.

## Worst-object convex decomposition follow-up

CoACD 1.0.11 produced a controlled 4/11/24/36/87-piece sweep for
`obj_size-large_ar-high_macro-high_rough-high`. Robust boolean-union auditing shows
that 87 pieces meet the provisional geometry gates (0.239/0.478/0.799 mm
p95/p99/max and 0.324% volume error), while 36 pieces are a close lower-cost option.
Macro-active p95 error falls from 2.894 mm for one hull to 0.206 mm at 87 pieces;
roughness-active p95 falls from 2.826 to 0.359 mm.

Geometry success did not translate to CPU representation fidelity. Deterministic
MuJoCo 3.3.1 N=500 hand and local-probe fixtures measured onset shifts versus the
original rigid flex:

| Fixture | Single hull | 11 pieces | 24 pieces | 36 pieces | 87 pieces |
|---|---:|---:|---:|---:|---:|
| Fingertip | -1.451 mm | -1.453 | -1.457 | -1.453 | -1.468 |
| Palm | -1.250 mm | -1.250 | -1.250 | -1.250 | -1.250 |
| Deepest concavity | +3.171 mm | +0.022 | -0.827 | -0.789 | -0.789 |
| Macro feature | -1.250 mm | -1.251 | -1.253 | -1.251 | -1.250 |
| Roughness feature | -1.250 mm | -1.253 | -1.253 | -1.253 | -1.251 |

The global late-contact shift identifies the unmapped 1.25 mm rigid-flex radius,
not a CoACD resolution limit. At a common pose 0.25 mm inside original-flex onset,
the bare decompositions have no fingertip, palm, macro, or roughness contact and
zero tactile response; the flex reference totals are 2.343, 2.057, 4.163, and
4.444. The 11-piece candidate alone passes the deepest-concavity fixture but fails
the other four.

The result is **Outcome C**: zero candidates are selected for Warp, so backend
parity, capacity, GPU throughput, complete-loop RL, all-24 generation, and training
smokes are not run. No production representation/default changed. A radius-aware
geometric shell or another native non-convex representation must pass the same CPU
fixtures first. Detailed contact points, normals, distances, forces, tactile errors,
top sensors, and region mapping are in
`generated/convex_decomposition_validation/contact_comparison.csv`,
`tactile_comparison.csv`, and `cpu_contact_tactile_validation.json`.

## Radius-aware rigid collision shell follow-up

The migration history is now:

```text
rigid-flex
  -> single rigid mesh (convexified by MuJoCo; Outcome C)
  -> bare convex decomposition (geometry improves, shell absent; Outcome C)
  -> radius-aware convex decomposition (shell restored locally; Outcome C)
```

MuJoCo 3.3.1 source confirms that each active rigid-flex tetrahedron collides with
support function `h_K(d) + radius`, mathematically `K + B(radius)`, independently
of its separate pair margin. The original worst object compiles at radius 1.25 mm,
zero margin, and zero gap. Repository generation scales the 1.0 mm rigid base radius
by object size: 0.75/1.0/1.25 mm for small/medium/large.

CPU MuJoCo 3.3.1 tested 61 unique candidates across the cached 11/36/87-piece
levels. The fixed grid covered 0.75, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, and 1.5 mm for
both margin control and geometric Minkowski expansion. A narrow 1.00-1.04 mm search
and 1.012/1.014 mm minimax check followed. Margin-gap diagnostics used 1.25 mm margin
with 0.10/0.25/0.50 mm gap.

| 36-piece geometric candidate | Fingertip | Palm | Deep concavity | Macro | Roughness | Worst |
|---|---:|---:|---:|---:|---:|---:|
| physical 1.250 mm | +0.028 | +0.023 | +0.500 | +0.014 | +0.006 | 0.500 |
| minimax 1.012 mm | -0.254 | -0.220 | +0.255 | -0.227 | -0.233 | 0.255 |

At 1.012 mm, common-pose tactile-total relative errors are 46.7% concavity, 100%
fingertip, 49.2% palm, 34.9% macro, and 47.0% roughness. Equal-relative-penetration
contact positions and normals remain within their declared gates, but force/tactile
totals do not: one rigid piece contact replaces contact patches of up to ten flex
contacts in the hand fixtures. Across all 610 candidate-fixture states, the explicit
0.1 mm / 5-degree diagnostic finds zero near-coincident duplicate-contact pairs.

The union-level geometry check uses the established Manifold3D boolean-union and
Trimesh/Rtree proximity pipeline with 100,000 deterministic samples. Its independent
target is the original non-convex surface Minkowski-summed with a 642-direction
1.25 mm sphere polytope. The physical 87-piece shell has 0.397% volume overfill and
0.246/0.493/0.816 mm p95/p99/max boundary error; 12.56% of target surface exceeds
0.10 mm error. The 1.012 mm minimax shell has 1.571% volume underfill and leaves
92.29% of target-surface samples outside its union. Thus the contact-calibrated
radius is not a geometrically faithful substitute for the physical radius.

This is **Outcome C**. No candidate passes the 0.10 mm primary onset gate, the
0.25 mm secondary diagnostic, and all contact/tactile gates. There are zero Warp
candidates; therefore CPU/Warp parity, capacity, GPU physics, complete-loop RL,
production selection, all-24 processing, study-factor checks, and N=500/N=1000 GPU
smokes were correctly not run. No TQC/HER/replay/normalization setting or production
collision default changed.

The exact next step is to investigate another native non-convex rigid collision
representation that preserves the source boundary before adding the size-scaled
physical shell. Durable results are in
`generated/convex_decomposition_validation/radius_aware_report.md`,
`radius_aware_parameter_sweep.json`, contact/tactile/duplicate CSVs, and the
explicit NOT_RUN CPU/Warp/GPU benchmark ledgers.

## Original rigid-flex CPU/Warp root-cause investigation

The exact original N=500 Gmsh rigid-flex model was isolated from the hand into a
single rigid flex plus a controlled sphere. CPU MuJoCo 3.3.1, CPU MuJoCo 3.11.0,
and MuJoCo Warp 3.11.0 compile the same 2387 vertices, 12602 edges, 8552
tetrahedra, 3328 shell triangles, 1.25 mm radius, and collision flags. All topology
hashes match; Warp world-position error is at most 1.86e-9 m.

The first disagreement occurs in Warp collision candidate generation. With the
sphere 20 mm outside the shell, both CPU versions report zero contacts. Warp 3.11
reports 2674 flex-internal contacts and 2674 constraint rows despite
`rigid=true`, `internal=false`, and `selfcollide=none`. The source cause is the
unconditional `_flex_tet_internal_collisions_detect` launch, whose kernel has no
rigid/internal inputs. A one-kernel diagnostic guard removes exactly all 2674
contacts and restores no-contact parity.

The external collision path is also non-equivalent. CPU uses the full
radius-inflated tetrahedron support with GJK/EPA; Warp 3.11 tests four inflated
triangle faces per tetrahedron and performs 1 mm position deduplication. Shallow
0.5 mm penetration matches at 3 contacts/12 rows and touch differs by only
2.12e-6, but at 2 mm CPU produces 13 contacts/52 rows versus guarded Warp's 10/40.
The full settled N=500 guarded step has 89 CPU versus 83 Warp contacts and retains
qpos 0.001607, qvel 0.803615, tactile maximum 74.3498, tactile RMSE 6.85656,
correlation 0.830962, active Jaccard 0.9, and 74.801% total tactile error. Thus the
internal guard is necessary but insufficient; the external manifold mismatch is
the material tactile failure.

Upstream commit `c822833` removed the exact internal kernel after v3.11.0, and
main `70c4571` fixes the separated state, but main still produces different
external contact counts. This is Outcome E and not production-ready. Complete
JSON/CSV evidence, a figure, full N=500 statistics, and the exact-asset reproducer
are under `generated/rigid_flex_cpu_warp_debug/`; the upstream-style package is
under `reproducers/rigid_flex_warp/`.

The focused root-cause suite passes 8/8. The complete CPU reference suite passes
94 tests with 11 expected skips, and `git diff --check` is clean.

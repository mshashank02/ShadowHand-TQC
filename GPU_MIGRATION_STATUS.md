# GPU Migration Status

Last updated: 2026-08-26

## Current phase

New native-rigid bare-CoACD experiment validated — **PASS_36**: the cached
36-piece representation passes the model, six-fixture, N=500 one-step, and
20-substep policy-action gates on matched MuJoCo/MuJoCo Warp 3.12.0; the cached
87-piece representation fails the 20-substep qpos/qvel gates. The production
default remains unchanged and full RL training has not started.

## Native-rigid CoACD 3.12 checkpoint: PASS_36

- Before making changes, reread `GPU_MIGRATION.md`, this status file, and every
  existing convex-decomposition validation report, JSON, CSV, cached-model
  manifest, and diagnostic figure. This study deliberately starts a new collision
  definition. Historical rigid-flex results are retained as history but are not
  reproduced, compared as a scientific gate, or pooled with these results.
- Reused the valid cached CoACD 1.0.11 decompositions; no decomposition asset was
  regenerated. The 36-piece cache key is
  `1c07a4d2506dd251a7ff42e6ad54106153f901726564845895959e7ed810c47e`, and the
  87-piece key is
  `b2a9781ce292e94cac9d05c48db9025e8bafe895c1dfe6ecb945493d4841fb4a`.
  Every cached piece hash was rechecked against its manifest.
- Both compiled models pass the requested structural contract: `nflex=0`, no
  `flexcomp`, original OBJ visual collision disabled, 36/87 convex mesh geoms on
  the single `object` body and `object:joint` free joint, and preserved body pose,
  mass `0.9765625`, COM, diagonal inertia, joint damping, friction, `condim`,
  `solref`, `solimp`, masks, and priority. Every object collision geom has
  `margin=0` and `gap=0`; the collision surface is the bare CoACD piece surface.
- Used CPU MuJoCo 3.12.0 and MuJoCo Warp 3.12.0 with Warp 1.16.0. MuJoCo Warp
  cannot transfer the unchanged hand model's existing non-zero-margin geom pair
  while MULTICCD is enabled, so MULTICCD and NATIVECCD were disabled identically
  in CPU and Warp. No object geom/contact parameter was changed.
- The separated control reports zero CPU/Warp object contacts and zero object
  constraint rows for both representations. Fingertip, palm, deepest-concavity,
  macro-feature, and roughness-feature poses were each defined 0.25 mm inside that
  representation's own CPU 3.12 onset, never relative to the historical flex.
  All ten contact fixtures pass: maximum absolute onset difference is
  `0.0000776 mm` for 36 pieces and `0.0005617 mm` for 87 pieces; maximum
  total-touch error is `0.5562%`; every active-sensor Jaccard is `1.0`; contacts
  and constraint rows agree at the evaluated poses; and no overflow occurs.
  Full positions, normals, forces, tactile totals, maximum errors, correlation,
  cosine similarity, active sets, and geom-pair multisets are stored in JSON, with
  the gate summary in CSV.
- Both CPU-settled, exactly matched N=500 one-step tests pass. For 36 pieces,
  qpos/qvel maximum errors are `1.1815e-6` / `5.9123e-4`, touch maximum error is
  `0.028983`, and total-touch error is `0.84721%`; CPU/Warp have 5/5 contacts and
  48/48 rows with identical geom-pair multisets. For 87 pieces the corresponding
  values are `1.7824e-6`, `8.7370e-4`, `0.00027897`, and `0.039516%`, with 4/4
  contacts and 45/45 rows. Both have zero overflow.
- The existing 20-substep policy-action gate separates the candidates. The
  36-piece result passes with qpos `6.5188e-5`, qvel `0.0028256`, touch maximum
  error `0.020793`, exact reward/success, matching 3/3 contacts and 39/39 rows,
  and zero overflow. The 87-piece result fails qpos (`6.8317e-4` versus the
  `3e-4` gate) and qvel (`0.028012` versus `0.02`), despite exact reward/success,
  matching 2/2 contact geom pairs and 35/35 rows, zero tactile error in this pose,
  and zero overflow. Three additional repeats confirm the 87-piece failure:
  qpos remains about `6.81e-4`–`6.89e-4` and qvel `0.0270`–`0.0343`.
- Geometry alone favors 87 pieces (p95/p99/max gap
  `0.239/0.478/0.799 mm`, volume error `0.324%`) over 36
  (`0.316/0.538/0.799 mm`, `0.443%`), but parity is the first selection criterion.
  The formal outcome is therefore **PASS_36**, not `PASS_BOTH`.
- Benchmarked only the parity-passing 36-piece representation, with CUDA graphs,
  20 physics substeps, and 5 measured task steps. At 1/16/64 worlds it delivers
  `48.87/541.78/1167.85` environment steps/s and uses approximately
  `186.2/354.0/991.5 MB` of total device memory at report time. All benchmark
  overflows are zero; maximum reported constraints/world are 55/84/99. The
  87-piece model was not benchmarked because its parity prerequisite failed.
- Durable machine-readable evidence is in
  `generated/native_decomposition_312_validation/validation_results.json`,
  `representation_36.json`, `representation_87.json`, `summary.csv`,
  `fixtures.csv`, and `gpu_benchmarks.csv`; `report.md` is the concise rendered
  conclusion. The production default was not changed, and full RL training was
  not run.

The passing 36-piece representation defines a new experiment. Before any full
study, all 24 objects and every sensor configuration must use the new definition
and be retrained; historical rigid-flex runs cannot be reused or pooled.

## Phase IV checkpoint: revised CPU 3.11/Warp 3.11 five-fixture and N=500 gates

- Following the user's explicit acceptance of the minimal 2 mm probe's 6.17%
  total-touch difference, changed the comparison reference to matching-version
  CPU MuJoCo 3.11.0 and set the revised total-touch tolerance to 6.5%. This does
  not reinterpret the separate CPU 3.3.1 compatibility result.
- Ran the unchanged fingertip, palm, deepest-concavity, macro-feature, and
  roughness-feature fixtures on the exact same 2D OBJ rigid-flex MJCF. Native
  MuJoCo 3.11 common-pose CPU/Warp total-touch errors are respectively 100.000%,
  7.667%, 24.761%, 14.379%, and 9.434%. All five exceed 6.5%, and every contact
  manifold differs. The fingertip is the most severe: onset shifts by -2.061 mm
  and CPU's 12 contacts/48 rows become Warp 0/0.
- Repeated the five fixtures with MULTICCD and NATIVECCD disabled in both
  backends, matching the configuration needed by the complete project model.
  The measured outcomes are effectively identical; no fixture passes.
- Native MuJoCo 3.11 defaults cannot load the complete N=500 model into MuJoCo
  Warp 3.11: Warp rejects the existing non-zero geom-pair margin while MULTICCD
  is enabled. The deployable full comparison therefore used MuJoCo 3.11.0 on
  both CPU and Warp with MULTICCD/NATIVECCD disabled; geometry, flex radius, and
  per-contact parameters were unchanged.
- The direct 2D CPU-settled control transfers with 99/99 contacts but has zero
  active tactile channels, so it cannot answer the tactile question. For the
  decisive test, settled the original 3D N=500 model for the established 200
  CPU steps, copied the identical qpos/qvel/control/warm-start state into the
  2D model, forwarded it, then transferred that active 2D state to Warp.
- At transfer, CPU and Warp have 101/101 contacts, 431/431 constraint rows, nine
  active sensors with Jaccard 1.0, total touch 527.817936 versus 527.817939,
  cosine 1.0, and negligible state error. This proves the state transfer itself
  is not the source of the later difference.
- After one matched step, CPU/Warp have 101/55 contacts and 431/247 constraint
  rows. Total touch is 527.817936 versus 736.288216, a 39.4966% relative error;
  maximum touch error is 128.3781, RMSE 8.21601, correlation 0.727854, cosine
  0.731162, and active Jaccard 0.9 (9 versus 10 active channels). Aggregate qpos
  and qvel maximum errors are 0.000509091 and 0.271492. Physical object errors
  are 0.019890 mm position, 0.033614 degrees orientation, 0.009935 linear
  velocity, and 0.293339 angular velocity.
- Classified the revised gate as a clear failure. The isolated 2 mm result was
  not representative of hand contact. Per the task ordering, N=1000, GPU and
  complete-loop benchmarks, all-24 preparation, and RL were not run.
- Durable evidence is under `generated/rigid_flex_2d_validation/`: the two
  five-fixture JSONs, `cpu_warp_2d_five_fixture_summary.csv`, the unseeded and
  active-seeded N=500 JSONs, revised `warp_decision.json`, and `final_report.md`.

Commands used the pinned `/tmp/shadowhand-mjw.revised-311` environment with CPU
MuJoCo 3.11.0, MuJoCo Warp 3.11.0, Warp 1.16.0, and CUDA. The five-fixture command
was `debug_rigid_flex_2d.py five-fixtures`; N=500 used
`diagnose_rigid_flex.py --fixture settled_contact`, including `--seed-xml` for
the active established state.

Final answer remains no: the 2D OBJ rigid flex is a valid CPU substitute and
avoids both tetrahedral Warp paths, but it does not provide matching-version
CPU/Warp contact or tactile parity on the exact fixtures or the full N=500 hand.

## Phase IV checkpoint: controlled Warp result and stop decision

- Built a minimal sphere-probe model from the same audited OBJ, 0.03125 scale,
  1.25 mm flex radius, explicit object mass/inertia/damping, and unchanged contact
  parameters. No tetrahedral guard or backend-specific MJCF was used.
- The 20 mm separated hard control passes: CPU MuJoCo 3.3.1, CPU MuJoCo 3.11.0,
  and stock MuJoCo Warp 3.11.0 each report zero contacts, zero constraint rows, and
  zero touch. This confirms that `dim=2` avoids the 2674-contact tetrahedral
  internal-kernel failure.
- A high-resolution normal-approach sweep brackets CPU and Warp onset at the same
  surface to Warp float resolution. The narrowest monotone Warp inside-contact
  sample is `7.62939453125e-10 m` inside the analytical surface; all outside
  samples remain contact-free.
- At 0.5 mm penetration, CPU 3.11.0 and Warp agree on 3 contacts/12 rows and touch
  differs by only `2.03e-6` (`0.896183136` versus `0.896181107`). CPU 3.3.1 also
  has 3/12 but touch is `0.312364754`, exposing a large MuJoCo 3.3→3.11 force-law
  change for 2D flex even when the contact manifold agrees.
- At the decisive 2 mm penetration, CPU 3.11.0 produces 13 contacts/52 rows and
  touch `9.719067013`; Warp produces 10 contacts/40 rows and touch `9.119109154`.
  The `0.599957859` touch difference and missing three contacts/twelve rows are a
  material deep-contact manifold failure. The same 10-contact Warp result observed
  after guarding the old 3D path is not repaired by supplying the surface directly.
- At this checkpoint the experiment was classified as **Outcome C — CPU 2D works,
  Warp still differs**, and the later phases were initially stopped. The user
  subsequently accepted a revised 6.5% matching-version tolerance and explicitly
  authorized the five-fixture and N=500 runs; their results are recorded in the
  newer checkpoint above. No new performance or training-speed claim is made.
- Added durable approach/contact/tactile CSVs, `warp_decision.json`, and
  `final_report.md` under `generated/rigid_flex_2d_validation/`, plus focused
  recorded-artifact and surface-semantics tests in
  `tests/test_rigid_flex_2d_validation.py`.

Commands: `debug_rigid_flex_2d.py prepare|cpu|warp` using CPU MuJoCo 3.3.1,
matching CPU MuJoCo 3.11.0, MuJoCo Warp 3.11.0, and Warp 1.16.0; then
`python -m object_conversion.report_rigid_flex_2d_validation`.

Checkpoint answer: the 2D OBJ rigid flex is a valid CPU representation under the
declared gates and does avoid the problematic tetrahedral path, but it cannot
replace the original model for GPU training because stock Warp still fails deeper
2D contact-manifold parity. The later revised tests above strengthen this result.

## Phase IV checkpoint: 2D flex semantics and CPU representation gate

- Recovered the complete Phase I–III investigation state and retained all failed
  native-mesh, convexity, convex-decomposition, radius-aware, and source-level
  root-cause history. No trainer, TQC/HER, production collision default, GJK/EPA,
  mesh decomposition, or study geometry was changed.
- Verified MuJoCo 3.3.1 and 3.11.0 `mjCFlexcomp::MakeMesh` semantics and MuJoCo
  Warp 3.11.0 `collision_flex.py`. OBJ faces compile directly to `dim=2`
  triangular flex elements. `rigid=true` assigns every vertex to the parent object
  body, preserving one free joint and six rigid-body DOFs. Radius remains the
  physical thickness of each triangle and the surface is not convexified.
- Confirmed source-level avoidance of both known 3D paths. The Warp
  `_flex_tet_internal_collisions_detect` kernel returns for every `dim != 3` flex,
  while `_flex_narrowphase_tet_detect` is separate from the 2D
  `_flex_narrowphase_unified` primitive-versus-triangle path.
- Reused the exact deterministic exterior OBJ without regeneration. Source GMSH
  SHA-256 is `eaa78c4a15423bf7346f120e1802f0201ee18711834cd703e6a4ef5f98e2b3ef`;
  OBJ SHA-256 is `e8fd23e33daf4bd61f71ee31d9e3ae8cdf5bd574e411fc608f5e9094def0d70c`.
  It has 1666 vertices, 3328 triangles, one watertight component, and is compiled
  with the unchanged 0.03125 scale and 1.25 mm radius.
- Added `object_conversion/validate_rigid_flex_2d.py`. It derives the full N=500
  2D model while preserving the object body, pose, explicit mass/COM/inertia,
  free-joint damping, target/reward contract, inherited contact settings, and all
  external assets. The compiled 2D flex has 1666 vertices, 4992 edges, 3328
  elements, `nq=38`, `nv=36`, one object free joint, and all flex vertices on the
  one object body.
- The 1/10/100-step contact-free CPU control is exact: zero object contacts and
  zero position, orientation, linear-velocity, and angular-velocity error at every
  checkpoint.
- Reused the exact fingertip, palm, deepest-concavity, macro-feature, and
  roughness-feature definitions and 28-iteration bisection. Maximum absolute
  onset error is `0.0000016764 mm`; contact and constraint-row counts match in all
  common and equal-relative-penetration states; maximum contact-position error is
  `0.018292 mm`; maximum normal error is below `0.00769 deg`; active tactile
  Jaccard is 1.0 throughout.
- Common-pose total tactile relative error is 1.708% fingertip, 7.938% palm,
  10.239% deepest concavity, 9.577% macro, and 9.971% roughness. These pass the
  predeclared 25% CPU representation gate; fingertip tactile correlation and
  cosine similarity both exceed 0.999997. The one-channel local probes have
  cosine 1.0 and no meaningful correlation statistic.
- CPU classification is **Outcome A — strong representation fidelity**. All
  primary onset/contact/tactile fixture gates pass, so the task's CPU hard gate
  authorizes the controlled MuJoCo Warp phases. This does not yet authorize N=1000,
  benchmarks, production integration, or RL.

Commands: `/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m
object_conversion.validate_rigid_flex_2d`; source inspection used the installed
MuJoCo 3.3.1/3.11.0 and MuJoCo Warp 3.11.0 trees. Durable artifacts are under
`generated/rigid_flex_2d_validation/`.

Exact next action: run the identical 2D MJCF through the 20 mm CPU/Warp
zero-contact control, then quasi-static onset, 0.5/2 mm contact, and the exact five
fixtures. Do not run N=1000 or performance work before N=500 parity passes.

## Phase III completed conclusion

Outcome E: MuJoCo Warp 3.11 first diverges by generating forbidden
tetrahedral-internal contacts, and its separate geom-flex triangle-face
narrowphase does not reproduce CPU's radius-inflated tetrahedron GJK/EPA manifold;
production remains flex-rejecting.

## Phase III checkpoint: exact-model source and transfer localization

- Preserved the existing radius-aware worktree and reread the migration/status,
  validation, convexity, decomposition, CPU contact/tactile, and radius-semantics
  artifacts before starting this investigation.
- Locked the version matrix to CPU MuJoCo 3.3.1, CPU MuJoCo 3.11.0, and MuJoCo
  Warp 3.11.0 with Warp 1.16.0 on CUDA.
- Recompiled the unchanged N=500 `flexcomp type="gmsh" dim="3" rigid="true"`
  model. It has 2387 vertices, 12602 edges, 8552 tetrahedra, a 1.25 mm radius,
  and one single-body rigid flex with `internal=false` and `selfcollide=none`.
- Verified CPU-to-Warp transfer preserves topology, radius, collision masks,
  `flex_internal=0`, and `flex_selfcollide=0`; Warp also derives
  `has_flex_selfcollide=False`.
- Compared the CPU 3.3.1 and 3.11.0 collision driver with MuJoCo Warp 3.11.0.
  Both CPU versions skip internal and self collision for `flex_rigid`, then gate
  the two paths by their respective flags. Warp instead launches
  `_flex_tet_internal_collisions_detect` whenever `nflexelem > 0`; that kernel
  receives neither `flex_rigid` nor `flex_internal` and therefore cannot honor
  either exclusion.
- This source path produces flex-only contact identifiers (`geom=-1`, same flex on
  both sides), matching the already measured 2674 false Warp contacts in the
  separated no-contact fixture. Compilation, transfer, radius omission, collision
  masks, CPU-version drift, trainer logic, and native-geom collision have been
  ruled out as the first divergence.

Exact next task: attribute the false-contact count to the tetrahedral kernel at
runtime, preserve real sphere-shell contacts under an isolated guard, and capture
the three-backend minimal reproduction.

## Phase III checkpoint: minimal reproducer and kernel counterfactual

- Added a focused diagnostic module/CLI and generated a minimal model containing
  only the exact original binary Gmsh asset, its unchanged rigid 3-D flex settings,
  and one position-controlled sphere probe. The compiled minimal model preserves
  all flex topology hashes and counts from the full N=500 model.
- Ran separated, approach, onset, shallow/deep penetration, and bidirectional
  sliding states in CPU MuJoCo 3.3.1, CPU MuJoCo 3.11.0, stock MuJoCo Warp 3.11.0,
  and an experimental Warp run that replaces only
  `_flex_tet_internal_collisions_detect` with a validated no-op.
- CPU 3.3.1 and 3.11.0 produce identical contact counts and distances in every
  state: zero for separated/approach/onset, three for shallow penetration, 13 for
  deep penetration, and four/seven for the two sliding states.
- Stock Warp adds exactly 2674 flex-internal contacts in every state, including the
  20 mm separated control. These contacts alone create 2674 constraint rows before
  any probe contact exists. Their distances range from about -0.001559 m to
  -2.81e-7 m.
- The one-kernel guard removes exactly those 2674 contacts and restores zero
  contacts/constraints in all separated states. It preserves real geom-flex
  collision: shallow penetration has three contacts on both CPU versions and
  guarded Warp, with minimum contact distance differing by about 1.1e-9 m.
- Guarded Warp still differs in manifold multiplicity for deeper/sliding states
  (10 versus 13 deep contacts and 3/2 versus 4/7 sliding contacts). That is a
  second narrowphase/manifold discrepancy and is the material remaining tactile
  failure after the earlier false-contact bug is removed.

First proven divergence: MuJoCo Warp 3.11 collision candidate generation launches
an intra-tetrahedron face/opposite-vertex test for a rigid flex that CPU MuJoCo
excludes. Constraints and tactile dynamics only amplify these already-invalid
contacts.

Exact next task: quantify the guard on the full N=500 settled tactile fixture,
inspect upstream history for this localized launch, and package the reproducer.

## Phase III checkpoint: full-model attribution, upstream status, and conclusion

- Inspected CPU `engine_collision_driver.c` and `engine_collision_convex.c` in
  MuJoCo 3.3.1 and 3.11.0, plus MuJoCo Warp 3.11 `io.py`, `smooth.py`,
  `collision_driver.py`, `collision_flex.py`, constraint, support, and sensor paths.
  CPU sends an active tetrahedron as one flex CCD object through `mjc_flexSupport`
  and GJK/EPA. Warp loops over all four triangle faces of every tetrahedron, uses
  analytic geom-triangle collision, and deduplicates positions within 1 mm; the
  source marks the missing tetrahedral geom-flex broadphase as a TODO.
- Refined the approach curve to +20/+2/+0.5/+0.1/0/-0.1/-0.25/-0.5/-1/-2 mm.
  CPU 3.3.1, CPU 3.11.0, and guarded Warp first contact at -0.1 mm with one
  contact/four rows. CPU-versus-guarded-Warp distance error is 7.1e-11 m. At
  -0.5 mm all have three contacts/12 rows and CPU 3.11 versus Warp touch differs
  by 2.12e-6. At -2 mm CPU has 13 contacts/52 rows and Warp has 10/40; touch
  differs by 0.777116.
- Ran the experimental guard on the full settled N=500 hand state. Imported data
  has 89 matched contacts, position error below 3e-8 m, touch max error 4.08e-6,
  and correlation 1.0. After one step CPU has 89 contacts and guarded Warp has 83;
  qpos/qvel/touch maximum errors remain 0.001607/0.803615/74.3498. Touch RMSE is
  6.85656, correlation 0.830962, active Jaccard 0.9, and total-magnitude relative
  error 74.801%. The guard changes the stock Warp contact count from 2757 to 83
  but does not make the external contact manifold scientifically equivalent.
- Searched upstream only after localization. The installed/tagged v3.11.0 commit
  is `dbc52e3`; commit `c822833` (`flex-flex collisions`, PR #1496, 2026-08-07)
  removes the exact internal kernel in a broader rewrite. Current main `70c4571`
  no longer produces false separated contacts, but measured shallow/deep counts
  are 4/16 versus CPU 3/13, so the rewrite is not complete CPU parity.
- Added the exact-asset reproducer, a minimal v3.11 internal-flag patch illustration,
  version/topology/world-position/no-contact/approach/contact/support/constraint/
  force/tactile artifacts, full N=500 statistics, an SVG figure, and the durable
  root-cause report under `generated/rigid_flex_cpu_warp_debug/` and
  `reproducers/rigid_flex_warp/`.
- Focused CPU-safe root-cause tests pass 8/8; the complete CPU reference suite
  passes 94 tests with 11 expected skips. `git diff --check` is clean. The
  experimental guard is confined to diagnostics, validates a single-body rigid
  3-D flex with both collision flags disabled, and is not reachable from the
  production backend.

Commands used include `debug_rigid_flex_cpu_warp.py prepare|cpu|warp`,
`diagnose_rigid_flex.py --fixture settled_contact --experimental-tet-guard`, and
`report_rigid_flex_root_cause.py`. Exact raw invocations and all output matrices
are represented by the CLI metadata and artifacts.

Conclusion: model compilation, topology, radius transfer, and world positions
match. The first divergence is forbidden Warp tetrahedron-internal candidate
generation. The substantial tactile discrepancy remains because Warp's external
3-D geom-flex narrowphase/manifold algorithm is not CPU MuJoCo's full inflated-
tetrahedron support/GJK/EPA algorithm. Touch aggregation agrees closely when the
manifold agrees and is not the root cause.

Exact next action: keep production flex rejection; validate a future upstream
release only after it resolves the `nJfe=0` edge write and passes this exact
no-contact, approach, manifold, force, and full N=500 tactile suite.

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

## Convexity audit checkpoint (2026-08-12)

### Phase completed

Completed the diagnostic convexity/collision-geometry audit of exactly the 24 rigid
objects in `study_objects/sphere_study_v1/manifest.csv`. No production physics,
generated runtime XML, trainer, TQC, HER, replay, environment, or study/GPBO code was
changed.

### Commands

```bash
python -m unittest tests.test_object_conversion tests.test_rigid_mesh_generation -v
python -m unittest tests.test_convexity_audit -v
MPLCONFIGDIR=/tmp/matplotlib-convexity OPENBLAS_NUM_THREADS=1 \
  python object_conversion/audit_convexity.py --all
```

The local execution harness imposed a roughly 30-second command boundary, so this
session checkpointed each object with `--object-id`, asserted all 24 used 100,000
samples, then assembled them with `--assemble-existing`. The ordinary `--all`
command remains the primary reusable interface.

### Method and validation

- Production-scaled exterior surfaces are compared directly with their SciPy/Qhull
  convex hulls; this is not a repeat of source-volume versus extracted-surface
  conversion validation.
- MuJoCo 3.11.0 directly compiled the representative 1,666-vertex/3,328-face OBJ
  with `mesh_graphadr=0` and a 989-vertex/1,974-face convex graph, exactly matching
  SciPy/Qhull. The CPU reference remains MuJoCo 3.3.1; production does not limit
  `maxhullvert`.
- Gap is the minimum normalized hull half-space slack, i.e. the shortest distance
  from an interior surface point to the convex hull boundary. Sampling is fixed-seed,
  deterministic, uniform by surface area, and uses 100,000 points/object.
- Every source vertex is checked separately. Sampled maxima are explicitly labeled
  estimates rather than continuous exact maxima.
- Synthetic convex-cube and known indented-cube controls pass. The concave control
  recovers exactly 20% volume inflation, the known 0.5-unit gap, and the correct
  maximum location.
- Worst-case 10k/50k/100k/250k p95, p99, and sampled maximum are stable; the 250k
  values are 2.545/3.427/4.460 mm.

### Summary measurements

- maximum volume inflation: **4.423%**;
- maximum 100k p95 gap: **2.551 mm**;
- maximum 100k p99 gap: **3.420 mm**;
- maximum 100k sampled gap: **4.459 mm**;
- maximum deterministic vertex gap: **4.459 mm**;
- largest normalized sampled maximum: **3.589%** of bounding-box diagonal;
- maximum surface area estimate above 0.1 mm: **41.47%**;
- worst object: `obj_size-large_ar-high_macro-high_rough-high`.

Macro-high is systematically more convexified than macro-low: mean p99 is
2.178 versus 0.227 mm and mean volume inflation is 3.078% versus 0.254%. The macro
distinction remains visible to collision (matched hull p95 differences
3.993-8.367 mm), but its concave component is suppressed. Roughness is encoded by
distinct geometry while rigid friction remains fixed; rough-high mean p99 is
1.566 versus 0.839 mm, so convexification directly affects that factor as well.

### Artifacts

- `object_conversion/audit_convexity.py`
- `tests/test_convexity_audit.py`
- `generated/convexity_audit/convexity_audit.csv`
- `generated/convexity_audit/convexity_audit.json`
- `generated/convexity_audit/convexity_audit.md`
- `generated/convexity_audit/converted_surfaces/`
- `generated/convexity_audit/figures/`
- summary appended to `generated/gpu_validation/validation_report.md`

Regression verification: the focused audit/conversion/generation set passed 16/16
tests, artifact integrity checks passed, and the final full CPU suite passed
**43/43** in the `ShadowHand` Conda environment. The base Python lacks PyTorch and
therefore cannot import four unrelated study test modules; those same modules pass
in the reference environment. The GPU suite was not required because the audit adds
standalone diagnostic code and does not alter shared or production GPU code.

### Conclusion and exact next step

This is **Outcome C**. Millimeter-scale gaps cover large surface regions and are
comparable to the 7.05 mm distal-finger collision radius and 2.5 mm tactile-site
half-thickness; the current single hull is not scientifically safe for the complete
factorial object study. Long-run multi-seed CPU/GPU learning validation is **not
approved as the next phase** for all 24 objects.

The exact next step is a separate, targeted representation study for the affected
macro-high and rough-high families: evaluate convex decomposition fidelity and cost,
then run controlled fingertip/contact comparisons. Do not silently replace the
production representation. Resume expensive learning validation only after the
collision model is shown to preserve the intended macro and roughness factors.

## Convex decomposition checkpoint: Phases A-B (2026-08-12)

### Completed work

- Selected **CoACD 1.0.11** after verifying its current Linux/Python support,
  deterministic seed, real-metric concavity threshold, hull-count limit, MIT
  license, and May 2026 release. VHACD 4.1 remains a BSD-3-Clause fallback, not the
  primary implementation.
- Selected **Manifold3D 3.5.2** for robust boolean unions, plus **Trimesh 5.0.0**
  and **Rtree 1.4.1** for exact closest-triangle queries. This prevents overlapping
  piece volume from being summed and prevents internal overlap faces from entering
  surface-distance metrics.
- Added `requirements-convex-decomposition.txt`. These dependencies are required
  only for offline asset generation/auditing; cached OBJ collision pieces do not
  add a trainer runtime dependency.
- Added `object_conversion/convex_decomposition.py`: fixed-seed, real-metric CoACD;
  full-parameter/source/exterior/scale/version cache keys; stable piece ordering;
  atomic cache writes; deterministic OBJ output in metres; per-piece SHA-256; and
  explicit watertightness, winding, convex-hull-volume, half-space, and index
  validation.
- Added `object_conversion/audit_decomposition.py`: Manifold Mesh64 boolean union,
  independent boundary-volume cross-check, closest-triangle surface distances, and
  exact union-membership classification from convex half-spaces. It reports signed
  volume error, absolute gap statistics, and separate overfill/underfill fractions.
- Added the checkpointable worst-object sweep driver
  `object_conversion/validate_convex_decomposition.py`. Each level writes its cache,
  union OBJ, metric arrays, and JSON independently before assembly. Heatmaps share
  the single-hull 4.459 mm physical colour range.
- CoACD pilots on `obj_size-large_ar-high_macro-high_rough-high`, seed 20260812,
  preprocessing off, real metric on, produced 4 pieces at a 4 mm threshold and 8
  pieces at 2 mm. All eight 2 mm pieces passed Manifold, and their boolean union was
  a valid 2,195-vertex/4,366-triangle boundary. These pilots chose the sweep scale;
  they are not yet final 100k-sample sweep results.

### Tests and commands

```bash
/tmp/shadowhand-mjw.7EqKBl/bin/pip install \
  coacd==1.0.11 manifold3d==3.5.2 trimesh==5.0.0 rtree==1.4.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/shadowhand-mjw.7EqKBl/bin/python -m unittest \
  tests.test_convex_decomposition tests.test_convexity_audit -v
```

The 13/13 tests pass. Synthetic controls prove deterministic cache reuse, reject a
non-convex shell, verify that overlapping boxes have boolean-union volume 1.5 rather
than summed volume 2.0, and show that an exact two-box concave union has essentially
zero error while its single hull retains a gap above 0.4 model units.

### Files changed

- `requirements-convex-decomposition.txt`
- `object_conversion/convex_decomposition.py`
- `object_conversion/audit_decomposition.py`
- `object_conversion/validate_convex_decomposition.py`
- `object_conversion/__init__.py`
- `tests/test_convex_decomposition.py`

No production MJCF generator, physics backend, TQC, HER, replay, or GPBO behavior
has changed in Phases A-B. Existing unrelated dirty worktree files were preserved.

### Exact next step

Phase C: add the explicit validation-only `single_mesh` versus
`convex_decomposition` generator selector; attach every piece to the existing
`object` body; retain exactly one free joint and the explicit inertial; retain the
original exterior as a non-colliding visual geom; copy identical contact parameters
to every piece; and add CPU compile/structure/cache-independence tests. Do not change
the default representation or run the full sweep until these invariants pass.

## Convex decomposition checkpoint: Phase C (2026-08-12)

### Completed work

- Added the explicit `single_mesh` / `convex_decomposition` selector to
  `pipeline_generate.py` and `generate_and_train.py`. `single_mesh` remains the
  default. Decomposition requires an explicit positive physical threshold; CoACD
  arguments are rejected when the decomposition representation is not selected.
- The decomposition branch retains the source exterior as `object_visual`, with
  `contype=0` and `conaffinity=0`, and attaches all transparent convex collision
  geoms directly to the existing `object` body. It creates no piece bodies or piece
  joints. Collision OBJ coordinates are metres and assets use scale `1 1 1`;
  the visual OBJ retains the production size scale.
- The existing `object:joint` free joint, body position, explicit inertial mass,
  centre of mass, and diagonal inertia remain authoritative. Every collision piece
  receives the unchanged friction, `condim`, `solref`, `solimp`, margin, gap,
  contype, conaffinity, and priority mapping.
- Extended representation manifests with visual/collision separation, CoACD and
  converter versions, full parameters, cache key/reuse, piece count/hashes/paths,
  units/scales, same-body assertion, and zero independent dynamic bodies. Legacy
  single-mesh manifest keys remain available.
- Generated a durable four-piece worst-object Phase-C fixture under
  `generated/convex_decomposition_validation/phase_c_model/` at threshold 4 mm.
  It compiles in both MuJoCo 3.3.1 and 3.11.0 with `nq=38`, `nv=36`, `nflex=0`,
  one visual object geom, four colliding object geoms, mass `0.9765625`, and
  diagonal inertia `0.0030517578125` on every axis. All four compiled contact
  parameter vectors are identical. MuJoCo 3.11 classifies it as GPU-supported
  `rigid_mesh_geom`.
- Generated a second candidate with a different sensor count and allocation. It
  reused the identical decomposition cache key
  `69e1b1a687c05f4cd5997ae9fe7e0b25378545b12d8fc4847a69b797b54df2b3`,
  identical four piece hashes, and no CoACD rerun. This demonstrates decomposition
  independence from N/alpha/beta.

### Verification

The focused conversion/decomposition/generator suite passes 14/14 tests in the
temporary validation environment. The new structural test also compiles its MJCF
and verifies one body, one free joint, explicit mass/inertia, one non-collision
visual, same-body collision pieces, unit piece scale, and exact contact mappings.

### Files changed

- `pipeline_generate.py`
- `generate_and_train.py`
- `tests/test_rigid_mesh_generation.py`
- Phase-C assets and manifests under
  `generated/convex_decomposition_validation/phase_c_model/`

No backend, TQC, HER, replay, or GPBO algorithm was changed. The default collision
representation was not switched.

### Exact next step

Phases D-E: run and checkpoint the five fixed-setting worst-object levels at 100,000
surface samples; measure the boolean union; assemble the direct single-hull table;
and generate same-scale heatmaps. Use the results, not nominal labels, to decide
whether the sweep spans the requested piece-count ranges and whether any candidates
meet the provisional geometry gates.

## Convex decomposition checkpoint: Phases D-F (2026-08-12)

### Completed work

- Completed five durable 100,000-sample worst-object levels with measured piece
  counts 4, 11, 24, 36, and 87. Exact results and cache keys are under
  `generated/convex_decomposition_validation/levels/` and
  `generated/rigid_mesh_cache/decomposition/`.
- The fixed CoACD search settings are seed 20260812, preprocessing off, real metric,
  resolution 1000, 20 MCTS nodes, 100 iterations, depth 3, merge on, no decimation,
  no extrusion, and 256 maximum vertices per piece. Thresholds are 4.0, 2.0, 1.0
  with a 24-piece cap, 1.4, and 1.2 mm respectively.
- An uncapped 1.0 mm diagnostic produced 164 pieces in 265.4 seconds. It is retained
  at `levels/uncapped_1mm/` but excluded from the practical 4-128 sweep. CoACD's
  `max_convex_hull` constrains the merge output rather than terminating the search;
  the 24-piece run therefore still took 243.3 seconds and emitted the expected
  warning that its capped maximum concavity exceeds the requested threshold.
- Assembled direct single-hull/decomposition CSV and JSON tables. Robust boolean
  unions range from 3,302 triangles at four pieces to 14,930 triangles at 87 pieces.
  All levels use identical source samples and the same distance method.
- Generated three-view, same-physical-scale heatmaps and coloured piece
  visualizations. Visual inspection confirms the 0-4.459 mm scale is shared and
  the bridged concavity progressively disappears.
- Added matched-factor region analysis. The strongest macro-change quartile has a
  9.190 mm p95 signal; p95 collision error falls from 2.894 mm for one hull to
  0.366 mm at 36 pieces and 0.206 mm at 87. The strongest roughness-change quartile
  has a 2.704 mm p95 signal; p95 error falls from 2.826 mm to 0.413 and 0.359 mm.

### Geometry table

| Representation | Pieces | Volume error | P95 mm | P99 mm | Max mm | >0.1 mm |
|---|---:|---:|---:|---:|---:|---:|
| Single hull | 1 | 4.423% | 2.551 | 3.420 | 4.459 | 41.47% |
| Very coarse | 4 | 1.769% | 1.011 | 1.977 | 3.164 | 31.45% |
| Coarse | 11 | 0.910% | 0.543 | 0.875 | 1.512 | 25.29% |
| Medium | 24 | 0.617% | 0.443 | 0.750 | 1.633 | 18.95% |
| Fine | 36 | 0.443% | 0.316 | 0.538 | 0.799 | 15.49% |
| Very fine | 87 | 0.324% | 0.239 | 0.478 | 0.799 | 11.73% |

The 87-piece candidate is the first practical-band candidate to pass all provisional
geometry targets. The 36-piece candidate is close and meets maximum/volume targets.
The 24-, 36-, and 87-piece representations proceed to CPU contact testing; 11 pieces
is retained as a coarse reference. This is not yet a production selection.

### Artifacts

- `worst_object_parameter_sweep.csv` and `.json`
- `geometry_comparison.csv`
- `feature_region_comparison.csv` and `.json`
- `convex_decomposition_report.md`
- `figures/worst_object_same_scale_gap_heatmaps.png`
- `figures/worst_object_decomposition_pieces.png`
- per-level union OBJ, arrays, JSON, and decomposition manifests/pieces

### Exact next step

Phases G-H: build the N=500 original rigid-flex, single-hull, 11-, 24-, 36-, and
87-piece CPU models with identical body/inertial/contact/sensor configuration. Run
deterministic isolated fingertip, palm, deepest-concavity onset, macro-feature, and
roughness-feature fixtures. Report first contact, contact point/normal/distance/force,
tactile max/mean/RMSE, active overlap, totals, top sensors, and region mapping. Only
then select 2-3 candidates for Warp.

## Convex decomposition checkpoint: Phases G-I (2026-08-12)

### Completed work

- Reused the already-generated exact N=500 original rigid-flex, single-hull, 11-,
  24-, 36-, and 87-piece models. No decomposition or model generation was repeated.
- Added deterministic source-local definitions for the 100k-sample deepest
  concavity, macro p95 feature, and roughness p95 feature, including source face and
  outward normal, at `contact_feature_definitions.json`.
- Added `object_conversion/validate_decomposition_contacts.py`. Isolated fingertip
  and palm fixtures use the exact 500 hand touch channels. The three local feature
  fixtures use a temporary validation-only 3 mm mocap sphere/site; it is compiled
  beside each source XML and deleted immediately, and never alters a production
  model or its N=500 contract.
- All fixtures run in CPU MuJoCo 3.3.1 with production gravity, zero initial
  velocity, identical explicit mass/inertia/contact properties, unrelated collision
  geoms disabled, 28-step onset bisection, and 0.25 mm response depth. They report
  first contact position/normal/distance/force, order-independent contact-patch
  errors, tactile max/mean/RMSE, activation overlap, totals, top sensors, and mapped
  hand region.
- Results are evaluated both at the same physical pose (0.25 mm inside original
  rigid-flex onset) and at equal penetration relative to each representation's own
  onset. This separates missing/early contact from solver response after contact.

### Contact-onset result

| Fixture | Single hull | 11 pieces | 24 pieces | 36 pieces | 87 pieces |
|---|---:|---:|---:|---:|---:|
| Fingertip shift vs flex | -1.451 mm | -1.453 | -1.457 | -1.453 | -1.468 |
| Palm shift vs flex | -1.250 mm | -1.250 | -1.250 | -1.250 | -1.250 |
| Deepest concavity shift | +3.171 mm | +0.022 | -0.827 | -0.789 | -0.789 |
| Macro-feature shift | -1.250 mm | -1.251 | -1.253 | -1.251 | -1.250 |
| Roughness-feature shift | -1.250 mm | -1.253 | -1.253 | -1.253 | -1.251 |

Positive is premature outward contact and negative is late contact. The single hull
demonstrates the expected 3.171 mm premature deepest-concavity contact. The 11-piece
candidate restores that local onset to 0.022 mm and passes every declared gate for
that fixture. It nevertheless fails the other four fixtures, as do all finer levels.

At the common original-flex pose, every bare decomposition has zero fingertip, palm,
macro, and roughness contacts/tactile response. Original rigid-flex tactile totals
are 2.343, 2.057, 4.163, and 4.444 respectively. Equal-penetration results are
retained separately and show active mapped finger/palm/probe signals, but cannot
erase the common-pose absence of contact.

### Scientific conclusion and gating decision

This is **Outcome C**. The dominant error is the already documented unmapped rigid-
flex radius: the original uses a 1.25 mm shell, while the bare convex meshes use the
extracted surface and zero margin/gap. The palm/macro/roughness onset errors reproduce
the missing radius almost exactly; the fingertip adds about 0.20 mm local error.
Increasing decomposition piece count cannot restore a shell absent from every piece.

No candidate passes all five predeclared CPU representation-fidelity gates. Phase I
therefore selects **zero Warp candidates**. CPU/Warp parity, capacity, GPU physics,
complete-loop RL, all-24 decomposition, N=500/N=1000 training smokes, production
default changes, and long multi-seed runs were intentionally not started.

### Artifacts and verification

- `object_conversion/validate_decomposition_contacts.py`
- `tests/test_decomposition_contact_validation.py`
- `generated/convex_decomposition_validation/contact_feature_definitions.json`
- `contact_comparison.csv`, `tactile_comparison.csv`
- `cpu_contact_tactile_summary.csv`, `.json` validation payload, and `.md` report
- updated `convex_decomposition_report.md`

The complete focused conversion/decomposition/contact/generation suite passes
**25/25** in the temporary validation environment. Source compilation and
`git diff --check` pass. The complete discovery suite in the `ShadowHand` reference
environment is green across **92 cases** (81 passed, 11 expected opt-in/optional-
dependency skips); no GPU run was appropriate after the CPU fidelity gate failed.

### Exact next step

Do not proceed to Warp. Evaluate an explicitly radius-aware rigid collision shell
(preferably a geometric offset that leaves the preserved margin/gap/contact solver
parameters unchanged), or another native non-convex rigid representation, on the
same five CPU fixtures. Only a candidate passing these representation-fidelity gates
may reopen Phase J.

## Radius-aware shell checkpoint: Phases A-D (2026-08-13)

### Completed phase

- Phase A: verified the exact MuJoCo 3.3.1 rigid-flex radius semantics from the
  tagged CPU engine source and the compiled reference model.
- Phase B: verified the repository's size-scaled radius rule for all 24 study
  objects: 0.75 mm small, 1.0 mm medium, and 1.25 mm large.
- Phase C: implemented separate margin-control and true convex Minkowski-shell
  model builders. Hybrid geometry plus residual margin is available but is not
  authorized unless the two primary strategies leave a measured residual.
- Phase D: added deterministic shell, convexity, radius-rule, and model-preservation
  tests. A generated 11-piece margin candidate compiles in CPU MuJoCo 3.3.1 with
  `nflex=0`, 59 geoms, 24 meshes, and all 529 compiled sensors.

### Exact commands

```text
pytest -q tests/test_radius_aware_collision.py tests/test_decomposition_contact_validation.py tests/test_rigid_mesh_generation.py
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.radius_aware_collision --models-manifest generated/convex_decomposition_validation/contact_models/models.json --output-dir /tmp/radius-aware-compile-check --levels coarse --strategy margin --shell-mm 1.25
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -c "import mujoco; m=mujoco.MjModel.from_xml_path('/tmp/radius-aware-compile-check/coarse_margin_1250um.xml'); print(mujoco.__version__,m.nflex,m.ngeom,m.nmesh,m.nsensor)"
```

The focused suite passes 12/12. The compile check reports
`3.3.1 0 59 24 529`.

### Files changed

- `object_conversion/radius_aware_collision.py`
- `object_conversion/validate_radius_aware_contacts.py`
- `tests/test_radius_aware_collision.py`
- `generated/convex_decomposition_validation/radius_semantics.md`
- `generated/convex_decomposition_validation/rigid_flex_radius_rule.csv`
- this status file

### Flex-radius finding and shell strategy

The flex support function adds `flex_radius + margin/2` in the normalized support
direction, so the active tetrahedron is `K + B(radius)`. A geom margin is retained
only as a diagnostic onset control because it splits pair inflation and is not
contact-point or tactile equivalent. The geometric candidate sums every convex
piece with a deterministic 162-direction sphere polytope, circumscribed to avoid
under-filling the true ball, then validates the resulting watertight convex hull.
Visual geometry, the one-body/one-free-joint structure, inertia, piece count, and
all non-margin contact parameters remain unchanged.

### Parameters and results so far

No five-fixture parameter result is claimed at this checkpoint. The intended first
sweep is 0.75, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, and 1.5 mm on the cached 11-, 36-,
and 87-piece levels. Primary onset tolerance is 0.10 mm; 0.25 mm remains a
secondary diagnostic. Tactile, duplicate-contact, CPU/Warp, and GPU benchmark
results are pending.

### Failures and exact next action

No implementation or compile failure is open. Run the margin diagnostic sweep on
all five CPU fixtures, then run the geometric-shell sweep (and only a targeted
hybrid residual if the measured data justify it). Do not run Warp unless one
candidate passes every CPU onset, contact, tactile, and duplicate-contact gate.

## Radius-aware shell checkpoint: Phases E-S (2026-08-13)

### Completed phases and selection

- Phases E-I: evaluated 61 unique margin-control and geometric Minkowski-shell
  candidates on the cached 11-, 36-, and 87-piece worst-object models using all
  five CPU fixtures, both common-physical-pose and equal-relative-penetration
  states, and explicit duplicate-contact diagnostics.
- Phase J: selected zero CPU-passing candidates. Classification is **Outcome C**.
- Phases K-O were not run because the mandatory CPU gate failed: no CPU/Warp
  parity, capacity, GPU physics, complete-loop RL, or production representation.
- Phases P-R were consequently not run: no all-24 conversion, factor-preservation
  approval, or N=500/N=1000 GPU training smoke.
- Phase S: consolidated durable reports/CSVs/JSON, tests, and migration docs. TQC,
  HER, replay, normalization, and production collision defaults remain unchanged.

### Exact CPU commands

```text
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.validate_radius_aware_contacts --strategy margin --shell-mm 0.75 1.0 1.1 1.2 1.25 1.3 1.4 1.5 --levels coarse fine very_fine --output generated/convex_decomposition_validation/radius_aware/margin_sweep
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.validate_radius_aware_contacts --strategy minkowski --shell-mm 0.75 1.0 1.1 1.2 1.25 1.3 1.4 1.5 --levels coarse fine very_fine --sphere-subdivisions 2 --sphere-bound circumscribed --output generated/convex_decomposition_validation/radius_aware/minkowski_sweep
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.validate_radius_aware_contacts --strategy minkowski --shell-mm 1.00 1.01 1.02 1.03 1.04 --levels fine very_fine --sphere-subdivisions 2 --sphere-bound circumscribed --output generated/convex_decomposition_validation/radius_aware/minkowski_refinement
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.validate_radius_aware_contacts --strategy minkowski --shell-mm 1.012 1.014 --levels fine --sphere-subdivisions 2 --sphere-bound circumscribed --output generated/convex_decomposition_validation/radius_aware/minkowski_minimax
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m object_conversion.validate_radius_aware_contacts --strategy margin --shell-mm 1.25 --gap-mm {0.10,0.25,0.50} --levels fine --output generated/convex_decomposition_validation/radius_aware/margin_gap_{0100,0250,0500}um
python -m object_conversion.report_radius_aware_validation --input <seven sweep JSON files> --output generated/convex_decomposition_validation
/tmp/shadowhand-mjw.7EqKBl/bin/python -m object_conversion.audit_radius_aware_geometry --reference-model generated/convex_decomposition_validation/contact_models/collision_fine_500_1.071429_0.714286/manipulate_collision_fine_touch_sensors_500_1.071429_0.714286.xml --candidate-manifest <11-piece-1.25 manifest> --candidate-manifest <36-piece-1.25 manifest> --candidate-manifest <87-piece-1.25 manifest> --candidate-manifest <36-piece-1.012 manifest> --output generated/convex_decomposition_validation --target-radius-mm 1.25 --samples 100000 --target-sphere-subdivisions 3
/home/mshashank02/anaconda3/envs/ShadowHand/bin/python -m pytest -q
```

The brace notation summarizes three separately executed gap commands; the exact
expanded commands and outputs are retained in the three named directories.

### Parameters tested and onset results

The fixed shell grid was 0.75/1.0/1.1/1.2/1.25/1.3/1.4/1.5 mm. The physical
1.25 mm 36-piece geometric shell gives onset shifts (candidate minus flex) of:

| Fingertip | Palm | Deep concavity | Macro | Roughness | Worst |
|---:|---:|---:|---:|---:|---:|
| +0.0284 mm | +0.0226 | +0.4999 | +0.0136 | +0.0064 | 0.4999 |

The narrow minimax refinement selects the 36-piece 1.012 mm geometric shell, but
it still gives -0.2536/-0.2197/+0.2545/-0.2273/-0.2335 mm across the same fixture
order. Its 0.2545 mm worst error fails the secondary 0.25 mm diagnostic and the
primary 0.10 mm gate. The 87-piece level is slightly worse at the fingertip. The
11-piece level already overfills the deepest concavity without a shell, so adding
the physical radius makes that local contact 1.294 mm premature.

At 1.25 mm margin, 0.10/0.25/0.50 mm gaps leave detected first-contact onset
unchanged. They reduce or eliminate active constraint force and tactile response,
so none improves fidelity. A hybrid uniform residual was not run because it adds
the same support offset to the same piece union and cannot reconcile the opposing
concavity/exterior errors. Piece-specific radii were rejected as a nonphysical fit
that would violate the one size-defined radius rule.

### Contact, force, tactile, and duplicate results

For the minimax 1.012 mm candidate, equal-penetration contact position errors are
0.005-0.506 mm and normal errors are at most 8.80 degrees, within their gates.
Nevertheless, common-pose tactile-total relative errors are 46.7% concavity, 100%
fingertip, 49.2% palm, 34.9% macro, and 47.0% roughness, all above the 25% gate.
Equal-penetration force/tactile relative errors remain about 31.6% concavity, 0.8%
fingertip force but 62.5% fingertip tactile total, 38.7% palm, 46.6% macro, and
57.8% roughness. The rigid candidates frequently produce one piece contact where
the flex reference produces six or ten element contacts.

No evaluated candidate/state contains a near-coincident duplicate-contact pair
under the declared 0.1 mm position / 5-degree normal diagnostic (maximum zero over
610 candidate-fixture states). Duplicate shell contacts are therefore not the
failure mode; onset inconsistency, flex/piece patch multiplicity, forces, and
tactile totals are.

### Shell union geometry

The established Manifold3D/Trimesh/Rtree pipeline constructed an independent target
by Minkowski-summing the original 1,666-vertex/3,328-face non-convex surface with a
642-direction circumscribed 1.25 mm sphere polytope (maximum radial approximation
excess 5.69 micrometres). Exact candidate-piece boolean unions were compared at
100,000 deterministic target-surface samples:

| Candidate | Volume error | p95 mm | p99 mm | max mm | overfill fraction | underfill fraction |
|---|---:|---:|---:|---:|---:|---:|
| 11-piece, 1.25 mm | +0.927% | 0.550 | 0.880 | 1.504 | 96.48% | 3.52% |
| 36-piece, 1.25 mm | +0.504% | 0.325 | 0.552 | 0.816 | 97.46% | 2.54% |
| 87-piece, 1.25 mm | +0.397% | 0.246 | 0.493 | 0.816 | 98.07% | 1.93% |
| 36-piece, 1.012 mm | -1.571% | 0.239 | 0.312 | 0.576 | 7.71% | 92.29% |

The physical 87-piece shell retains 12.56% of target surface above 0.10 mm error
(4.87% above 0.25 mm), while the contact-minimax radius globally underfills the
physical target. These measurements are in `radius_aware_shell_geometry.csv/.json`.

### CPU-to-Warp and GPU results

There are no CPU-passing candidates, so CPU/Warp parity, contact/constraint
capacity, GPU physics throughput, complete-loop throughput, and learning were
**NOT RUN**. The three required CSV ledgers record `NOT_RUN_CPU_GATE_FAILED` rather
than fabricating empty benchmark results.

### Files changed and durable artifacts

- implementation: `object_conversion/radius_aware_collision.py`,
  `validate_radius_aware_contacts.py`, `report_radius_aware_validation.py`
- tests: `tests/test_radius_aware_collision.py` and the extended contact validator test
- semantics/rule: `radius_semantics.md`, `rigid_flex_radius_rule.csv`
- required results: `radius_aware_parameter_sweep.csv/.json`,
  `radius_aware_contact_comparison.csv`, `radius_aware_tactile_comparison.csv`,
  `duplicate_contact_analysis.csv`, `radius_aware_report.md`
- shell geometry: `radius_aware_shell_geometry.csv/.json`
- gated ledgers: `cpu_warp_radius_aware_parity.csv`,
  `radius_aware_gpu_benchmark.csv`, `radius_aware_complete_loop_benchmark.csv`
- exact generated XML/assets/manifests and per-sweep raw JSON under
  `generated/convex_decomposition_validation/radius_aware/`
- updated `GPU_MIGRATION.md`, `GPU_MIGRATION_STATUS.md`, and
  `generated/gpu_validation/validation_report.md`

The focused suite passes 14/14 in both the local and CPU reference environments.
The complete CPU reference suite passes **87 tests with 11 expected skips**.
`git diff --check` passes.

### Failures and exact next action

The physical shell cannot correct decomposition-local under/overfill consistently,
and matching onset does not reproduce flex contact-patch/tactile behavior. Do not
proceed with this convex-piece shell in Warp. Investigate a different native
non-convex rigid representation that preserves the original source boundary before
applying the 0.75/1.0/1.25 mm size-scaled collision shell, then rerun the same CPU
gates before any GPU work.

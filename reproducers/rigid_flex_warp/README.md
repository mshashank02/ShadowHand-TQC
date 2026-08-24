# MuJoCo Warp 3.11 rigid 3-D flex collision reproducer

This reproducer uses the project's exact original binary Gmsh asset rather than a
replacement shape. Generate the self-contained XML and asset copy with:

```bash
/tmp/shadowhand-mjw.7EqKBl/bin/python debug_rigid_flex_cpu_warp.py prepare \
  --source-xml generated/smoke_tests/rigid_n0500_a0p3_b0p6_large_high_high_high/custom_obj_size_large_ar_high_macro_high_rough_high_500_1.071429_0.714286/manipulate_custom_obj_size_large_ar_high_macro_high_rough_high_touch_sensors_500_1.071429_0.714286.xml \
  --source-msh generated/smoke_tests/rigid_n0500_a0p3_b0p6_large_high_high_high/stls/hand/obj_size_large_ar_high_macro_high_rough_high_obj_size-large_ar-high_macro-high_rough-high.msh \
  --output-dir generated/rigid_flex_cpu_warp_debug/minimal_reproducer
```

Then run the CPU and Warp collectors as documented by
`debug_rigid_flex_cpu_warp.py --help`. The 20 mm separated state gives:

| backend | contacts | flex-internal contacts | constraint rows |
|---|---:|---:|---:|
| CPU MuJoCo 3.3.1 | 0 | 0 | 0 |
| CPU MuJoCo 3.11.0 | 0 | 0 | 0 |
| MuJoCo Warp 3.11.0 | 2674 | 2674 | 2674 |
| Warp 3.11.0 with the isolated kernel guard | 0 | 0 | 0 |

The v3.11.0 source launches `_flex_tet_internal_collisions_detect` for every
3-D flex element without passing `flex_internal` or `flex_rigid`. CPU MuJoCo
skips this work for rigid flexes and gates it on `flex_internal` otherwise.
[`candidate_internal_flag_guard.diff`](candidate_internal_flag_guard.diff) shows
the smallest source-level guard for this tag; it is evidence, not a production
vendoring recommendation.

Upstream commit `c822833` (PR #1496, “flex-flex collisions”) later removed the
exact kernel in a broader collision rewrite. The current main branch fixes this
no-contact symptom but still does not reproduce CPU's deeper geom-flex contact
manifold, because v3.11.0 uses per-triangle tests rather than CPU's full
radius-inflated tetrahedron GJK/EPA path.

See `generated/rigid_flex_cpu_warp_debug/root_cause_report.md` for the complete
measurements.

import json
import tempfile
import unittest
from pathlib import Path

from study_common import (
    build_candidate_grid,
    load_score_from_artifacts,
    load_cluster_config,
    load_study_manifest,
    sobol_initial_candidates,
)


class StudyCommonTests(unittest.TestCase):
    def test_candidate_grid_has_expected_size(self):
        grid = build_candidate_grid()
        self.assertEqual(len(grid), 486)
        self.assertEqual(len({candidate.candidate_id for candidate in grid}), 486)

    def test_sobol_initial_candidates_are_unique_and_on_grid(self):
        candidates = sobol_initial_candidates(12)
        allowed_ids = {candidate.candidate_id for candidate in build_candidate_grid()}
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 12)
        self.assertTrue(all(candidate.candidate_id in allowed_ids for candidate in candidates))

    def test_load_study_manifest_validates_expected_combos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            rows = [
                ("obj_a_low_small", "obj_a_low_small.msh", "obj_a", "low", "small"),
                ("obj_a_low_large", "obj_a_low_large.msh", "obj_a", "low", "large"),
                ("obj_a_high_small", "obj_a_high_small.msh", "obj_a", "high", "small"),
                ("obj_a_high_large", "obj_a_high_large.msh", "obj_a", "high", "large"),
                ("obj_b_low_small", "obj_b_low_small.msh", "obj_b", "low", "small"),
                ("obj_b_low_large", "obj_b_low_large.msh", "obj_b", "low", "large"),
                ("obj_b_high_small", "obj_b_high_small.msh", "obj_b", "high", "small"),
                ("obj_b_high_large", "obj_b_high_large.msh", "obj_b", "high", "large"),
            ]
            manifest_path.write_text(
                "object_id,msh_file,base_object,aspect_ratio,size\n"
                + "\n".join(",".join(row) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            for _, mesh_name, *_ in rows:
                (root / mesh_name).write_text("mesh", encoding="utf-8")

            objects = load_study_manifest(str(root), expected_base_objects=2)
            self.assertEqual(len(objects), 8)
            self.assertEqual(objects[0].aspect_ratio, "high")

    def test_load_study_manifest_accepts_medium_size_when_present_consistently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            rows = [
                ("obj_a_low_small", "obj_a_low_small.msh", "obj_a", "low", "small"),
                ("obj_a_low_medium", "obj_a_low_medium.msh", "obj_a", "low", "medium"),
                ("obj_a_low_large", "obj_a_low_large.msh", "obj_a", "low", "large"),
                ("obj_a_high_small", "obj_a_high_small.msh", "obj_a", "high", "small"),
                ("obj_a_high_medium", "obj_a_high_medium.msh", "obj_a", "high", "medium"),
                ("obj_a_high_large", "obj_a_high_large.msh", "obj_a", "high", "large"),
                ("obj_b_low_small", "obj_b_low_small.msh", "obj_b", "low", "small"),
                ("obj_b_low_medium", "obj_b_low_medium.msh", "obj_b", "low", "medium"),
                ("obj_b_low_large", "obj_b_low_large.msh", "obj_b", "low", "large"),
                ("obj_b_high_small", "obj_b_high_small.msh", "obj_b", "high", "small"),
                ("obj_b_high_medium", "obj_b_high_medium.msh", "obj_b", "high", "medium"),
                ("obj_b_high_large", "obj_b_high_large.msh", "obj_b", "high", "large"),
            ]
            manifest_path.write_text(
                "object_id,msh_file,base_object,aspect_ratio,size\n"
                + "\n".join(",".join(row) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            for _, mesh_name, *_ in rows:
                (root / mesh_name).write_text("mesh", encoding="utf-8")

            objects = load_study_manifest(str(root), expected_base_objects=2)
            self.assertEqual(len(objects), 12)
            self.assertEqual({obj.size for obj in objects}, {"small", "medium", "large"})

    def test_load_study_manifest_accepts_explicit_native_rigid_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "manifest.csv").write_text(
                "object_id,native_task,base_object,aspect_ratio,size\n"
                "builtin_block,block,block,high,medium\n",
                encoding="utf-8",
            )

            objects = load_study_manifest(str(root), expected_base_objects=1)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].native_task, "block")
            self.assertEqual(objects[0].msh_file, "")
            self.assertEqual(objects[0].abs_msh_path, "")

    def test_manifest_row_cannot_mix_mesh_and_native_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "manifest.csv").write_text(
                "object_id,msh_file,native_task,base_object,aspect_ratio,size\n"
                "ambiguous,obj.msh,block,obj,high,medium\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_study_manifest(str(root), expected_base_objects=1)

    def test_gpu_metrics_schema_is_consumed_without_backend_specific_logic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "backend": "mujoco_warp",
                        "tasks": ["builtin_block"],
                        "checkpoints": [0.5, 1.0],
                        "success": {"builtin_block": [0.25, 0.75]},
                        "final_success": {"builtin_block": 0.75},
                    }
                ),
                encoding="utf-8",
            )
            score = load_score_from_artifacts(str(metrics_path))
            self.assertIsNotNone(score)
            self.assertGreater(score, 0.0)

    def test_load_cluster_config_computes_repo_path_and_python_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cluster_hosts.yaml"
            config_path.write_text(
                """
coordinator:
  host: coordinator
  ssh_target: user@coordinator
  gpu_count: 0
  work_root: /tmp/work
  python_root: /tmp/work/envs/shadowhand
  priority: 100
hosts:
  - host: pc1
    ssh_target: user@pc1
    gpu_count: 4
    work_root: /tmp/work
    python_root: /tmp/work/envs/shadowhand
    priority: 90
""".strip(),
                encoding="utf-8",
            )
            cluster_cfg = load_cluster_config(str(config_path), repo_dirname="ShadowHand-TQC")
            self.assertEqual(cluster_cfg.coordinator.repo_path, "/tmp/work/ShadowHand-TQC")
            self.assertTrue(cluster_cfg.coordinator.python_bin.endswith("/bin/python"))
            self.assertEqual(cluster_cfg.hosts[0].host, "pc1")
            self.assertIsNone(cluster_cfg.hosts[0].cpu_cores)
            self.assertEqual(cluster_cfg.hosts[0].resolved_num_envs_per_job(), None)

    def test_coordinator_can_also_be_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cluster_hosts.yaml"
            config_path.write_text(
                """
coordinator:
  host: pc2
  ssh_target: user@pc2
  gpu_count: 2
  work_root: /tmp/work
  python_root: /tmp/work/envs/shadowhand
  priority: 100
  role: coordinator
  run_worker: true
hosts:
  - host: pc1
    ssh_target: user@pc1
    gpu_count: 4
    work_root: /tmp/work
    python_root: /tmp/work/envs/shadowhand
    priority: 90
""".strip(),
                encoding="utf-8",
            )
            cluster_cfg = load_cluster_config(str(config_path), repo_dirname="ShadowHand-TQC")
            worker_names = [host.host for host in cluster_cfg.worker_hosts()]
            self.assertEqual(worker_names, ["pc2", "pc1"])

    def test_load_cluster_config_reads_num_envs_per_job_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cluster_hosts.yaml"
            config_path.write_text(
                """
coordinator:
  host: coordinator
  ssh_target: user@coordinator
  gpu_count: 2
  cpu_cores: 12
  num_envs_per_job: 5
  work_root: /tmp/work
  python_root: /tmp/work/envs/shadowhand
  priority: 100
hosts:
  - host: pc1
    ssh_target: user@pc1
    gpu_count: 4
    cpu_cores: 64
    num_envs_per_job: 15
    work_root: /tmp/work
    python_root: /tmp/work/envs/shadowhand
    priority: 90
""".strip(),
                encoding="utf-8",
            )
            cluster_cfg = load_cluster_config(str(config_path), repo_dirname="ShadowHand-TQC")
            self.assertEqual(cluster_cfg.coordinator.resolved_num_envs_per_job(), 5)
            self.assertEqual(cluster_cfg.hosts[0].resolved_num_envs_per_job(), 15)


if __name__ == "__main__":
    unittest.main()

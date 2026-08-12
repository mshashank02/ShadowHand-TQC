import tempfile
import unittest
from pathlib import Path

from optimize_dataset_gpbo import choose_next_candidate, validate_study_trainer
from study_common import Candidate, StudyObject
from study_queue import StudyQueue


class OptimizeDatasetGPBOTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "study.db")
        self.queue = StudyQueue(self.db_path)
        self.queue.register_hosts(
            [
                {
                    "host": "pc1",
                    "ssh_target": "user@pc1",
                    "gpu_count": 4,
                    "work_root": "/tmp/work",
                    "python_root": "/tmp/work/envs/shadowhand",
                    "repo_path": "/tmp/work/ShadowHand-TQC",
                    "python_bin": "/tmp/work/envs/shadowhand/bin/python",
                    "priority": 100,
                    "role": "worker",
                }
            ]
        )
        self.initial_candidate = Candidate(candidate_id="n0010_a0p1_b0p1", N=10, alpha=0.1, beta=0.1)
        self.available_candidate = Candidate(candidate_id="n0050_a0p2_b0p2", N=50, alpha=0.2, beta=0.2)
        self.queue.register_candidates([self.initial_candidate, self.available_candidate])
        self.spec = {
            "study_name": "test",
            "objects_root_relpath": "study_objects/test",
            "base_xml_relpath": "assets/hand_base.xml",
            "trainer_args": [],
            "seed": 0,
            "eval_episodes": 10,
            "force": False,
            "initial_candidate_ids": [self.initial_candidate.candidate_id],
            "bo_physics_modes": ["deformable", "rigid"],
            "sobol_physics_modes": ["rigid"],
        }

    def tearDown(self):
        self.queue.close()
        self.tempdir.cleanup()

    def test_bo_waits_for_initial_candidate_to_gain_required_physics_modes(self):
        rigid_job = {
            "object_id": "obj_a_low_small",
            "physics_mode": "rigid",
            "base_object": "obj_a",
            "aspect_ratio": "low",
            "size": "small",
            "seed": 0,
            "priority": 1,
            "artifact_relpath": "generated/test/rigid",
            "metrics_relpath": "generated/test/rigid/metrics.json",
            "stdout_relpath": "generated/test/rigid/stdout.txt",
            "stderr_relpath": "generated/test/rigid/stderr.txt",
            "payload": {"candidate_id": self.initial_candidate.candidate_id},
        }
        deformable_job = {
            "object_id": "obj_a_low_small",
            "physics_mode": "deformable",
            "base_object": "obj_a",
            "aspect_ratio": "low",
            "size": "small",
            "seed": 0,
            "priority": 0,
            "artifact_relpath": "generated/test/deformable",
            "metrics_relpath": "generated/test/deformable/metrics.json",
            "stdout_relpath": "generated/test/deformable/stdout.txt",
            "stderr_relpath": "generated/test/deformable/stderr.txt",
            "payload": {"candidate_id": self.initial_candidate.candidate_id},
        }

        self.queue.enqueue_candidate_jobs(self.initial_candidate, [rigid_job], source="sobol")
        self.queue.worker_heartbeat("pc1", "worker-1", 12345, details={})
        claimed = self.queue.claim_job("pc1", "worker-1", 0)
        self.assertIsNotNone(claimed)
        self.queue.finish_job(int(claimed["job_id"]), 0.4)

        candidate, source = choose_next_candidate(self.queue, self.spec)
        self.assertIsNone(candidate)
        self.assertIsNone(source)

        self.queue.enqueue_candidate_jobs(self.initial_candidate, [deformable_job], source="sobol")
        claimed = self.queue.claim_job("pc1", "worker-1", 1)
        self.assertIsNotNone(claimed)
        self.queue.finish_job(int(claimed["job_id"]), 0.6)

        candidate, source = choose_next_candidate(self.queue, self.spec)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.candidate_id, self.available_candidate.candidate_id)
        self.assertEqual(source, "bo")

    def test_gpu_study_accepts_rigid_geoms_and_requires_explicit_memory_policy(self):
        native = StudyObject(
            object_id="builtin_block",
            msh_file="",
            base_object="block",
            aspect_ratio="high",
            size="medium",
            abs_msh_path="",
            native_task="block",
        )
        args = [
            "--num-envs", "64",
            "--contacts-per-world", "128",
            "--constraints-per-world", "256",
            "--auto-replay-capacity",
        ]
        validate_study_trainer(
            "gpu",
            study_objects=[native],
            sobol_physics_modes=["rigid"],
            bo_physics_modes=["rigid"],
            trainer_args=args,
        )

        mesh = StudyObject(
            object_id="mesh_object",
            msh_file="object.msh",
            base_object="object",
            aspect_ratio="high",
            size="medium",
            abs_msh_path="/tmp/object.msh",
        )
        validate_study_trainer(
            "gpu",
            study_objects=[mesh],
            sobol_physics_modes=["rigid"],
            bo_physics_modes=["rigid"],
            trainer_args=args,
        )

        with self.assertRaisesRegex(ValueError, "top-level --trainer"):
            validate_study_trainer(
                "cpu",
                study_objects=[native],
                sobol_physics_modes=["rigid"],
                bo_physics_modes=["rigid"],
                trainer_args=["--trainer", "gpu"],
            )


if __name__ == "__main__":
    unittest.main()

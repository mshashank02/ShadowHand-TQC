import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from shadowhand_gpu.rl.normalization import CudaVecNormalize
from shadowhand_gpu.rl.replay import CudaHERReplayBuffer, plan_replay_memory, shadowhand_sparse_reward
from shadowhand_gpu.rl.tqc import TQCConfig, TQCLearner
from shadowhand_gpu.task import PerWorldRandom, ShadowHandTaskConfig, ShadowHandTaskStep
from shadowhand_gpu.trainer import (
    AutoTuneRecommendation,
    CudaTQCTrainer,
    TrainerConfig,
    load_complete_loop_recommendation,
    load_num_envs_recommendation,
    reference_gradient_steps,
    study_metrics_payload,
)


OBSERVATION_SHAPES = {
    "achieved_goal": (7,),
    "desired_goal": (7,),
    "observation": (3,),
}


class _FakeBackend:
    def __init__(self):
        self.model = SimpleNamespace(nu=20)
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class _FakeTask:
    def __init__(self, seed=0):
        self.worlds = 2
        self.device = torch.device("cpu")
        self.backend = _FakeBackend()
        self.config = ShadowHandTaskConfig(
            max_episode_steps=2,
            physics_steps_per_action=1,
            base_reset_settle_policy_steps=1,
        )
        self.observation_width = 3
        self.state = torch.zeros(self.worlds, dtype=torch.float32)
        self.episode_steps = torch.zeros(self.worlds, dtype=torch.int64)
        self.episode_ids = torch.full((self.worlds,), -1, dtype=torch.int64)
        self.rng = PerWorldRandom(self.worlds, seed, self.device)

    def observations(self):
        achieved = torch.zeros(self.worlds, 7)
        achieved[:, 0] = self.state
        achieved[:, 3] = 1.0
        desired = torch.zeros_like(achieved)
        desired[:, 3] = 1.0
        observation = torch.stack(
            (
                self.state,
                self.episode_steps.to(torch.float32),
                1.0 - self.episode_steps.to(torch.float32) / 2.0,
            ),
            dim=1,
        )
        return {
            "observation": observation,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool)
        self.state[mask] = 0.0
        self.episode_steps[mask] = 0
        self.episode_ids[mask] += 1
        self.rng.uniform(1, mask=mask)
        return self.observations()

    def step(self, actions):
        self.state.add_(1.0)
        self.episode_steps.add_(1)
        observations = self.observations()
        rewards = -torch.ones(self.worlds)
        truncated = self.episode_steps >= self.config.max_episode_steps
        terminated = torch.zeros_like(truncated)
        return ShadowHandTaskStep(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            success=torch.zeros_like(rewards),
            conditioned_actions=actions,
        )

    def compute_her_rewards(self, achieved, desired):
        return shadowhand_sparse_reward(achieved, desired)

    def checkpoint(self):
        return {
            "state": self.state.clone(),
            "episode_steps": self.episode_steps.clone(),
            "episode_ids": self.episode_ids.clone(),
            "rng_state": self.rng.state.clone(),
        }

    def load_checkpoint(self, checkpoint):
        self.state.copy_(checkpoint["state"])
        self.episode_steps.copy_(checkpoint["episode_steps"])
        self.episode_ids.copy_(checkpoint["episode_ids"])
        self.rng.state.copy_(checkpoint["rng_state"])


def _build_trainer(seed=4):
    torch.manual_seed(seed)
    task = _FakeTask(seed)
    normalizer = CudaVecNormalize(OBSERVATION_SHAPES, num_envs=2, device="cpu")
    learner = TQCLearner(
        TQCConfig(
            observation_dim=17,
            action_dim=20,
            hidden_dims=(8,),
            n_quantiles=3,
            top_quantiles_to_drop_per_critic=1,
            device="cpu",
        )
    )
    replay = CudaHERReplayBuffer(
        requested_capacity=16,
        num_envs=2,
        observation_shapes=OBSERVATION_SHAPES,
        action_dim=20,
        max_episode_steps=2,
        reward_function=task.compute_her_rewards,
        device="cpu",
    )
    return CudaTQCTrainer(
        task=task,
        normalizer=normalizer,
        replay=replay,
        learner=learner,
        config=TrainerConfig(batch_size=4, learning_starts=0),
        seed=seed,
    )


class TrainerTests(unittest.TestCase):
    def test_rollout_stores_terminal_observation_and_updates_after_complete_episode(self):
        trainer = _build_trainer()
        first = trainer.collect_and_update()
        self.assertIsNone(first.update_metrics)
        second = trainer.collect_and_update()
        self.assertIsNotNone(second.update_metrics)
        self.assertEqual(trainer.global_steps, 4)
        self.assertEqual(trainer.completed_episodes, 2)
        self.assertEqual(trainer.learner.updates, 1)
        self.assertEqual(trainer.replay.episode_length[:2].tolist(), [[2, 2], [2, 2]])
        # Row one is the terminal state (2), not the reset state (0).
        self.assertEqual(trainer.replay.next_observations["observation"][1, 0, 0], 2.0)
        # Initial reset, ordinary next observation, then auto-reset observation.
        self.assertAlmostEqual(float(trainer.normalizer.observation_rms["observation"].count), 6.0001)

    def test_full_checkpoint_resume_reproduces_next_stochastic_update(self):
        trainer = _build_trainer()
        trainer.collect_and_update()
        trainer.collect_and_update()
        with tempfile.TemporaryDirectory(prefix="shadowhand-trainer-test.") as root:
            path = trainer.save_checkpoint(Path(root) / "checkpoint.pt", include_replay=True)
            expected_step = trainer.collect_and_update()
            expected_actor = {
                key: value.detach().clone() for key, value in trainer.learner.actor.state_dict().items()
            }
            expected_replay_action = trainer.replay.actions[2].clone()

            resumed = _build_trainer()
            self.assertTrue(resumed.load_checkpoint_file(path))
            actual_step = resumed.collect_and_update()

        torch.testing.assert_close(actual_step.actions, expected_step.actions, rtol=0.0, atol=0.0)
        torch.testing.assert_close(resumed.replay.actions[2], expected_replay_action, rtol=0.0, atol=0.0)
        for key, value in resumed.learner.actor.state_dict().items():
            torch.testing.assert_close(value, expected_actor[key], rtol=0.0, atol=0.0)

    def test_bufferless_resume_resets_episode_and_waits_for_fresh_complete_history(self):
        trainer = _build_trainer()
        trainer.collect_and_update()
        checkpoint = trainer.checkpoint(include_replay=False)

        resumed = _build_trainer()
        self.assertFalse(resumed.load_checkpoint(checkpoint, resume_warmup_steps=0))
        self.assertEqual(resumed.rollout_episode_step, 0)
        self.assertEqual(resumed.replay.pos, 0)
        self.assertEqual(resumed.learning_block_until, 6)
        self.assertIsNone(resumed.collect_and_update().update_metrics)
        self.assertIsNone(resumed.collect_and_update().update_metrics)
        self.assertIsNotNone(resumed.collect_and_update().update_metrics)

    def test_replay_plan_rejects_capacity_smaller_than_one_episode(self):
        with self.assertRaisesRegex(ValueError, "complete episode"):
            plan_replay_memory(
                requested_capacity=3,
                num_envs=2,
                observation_shapes=OBSERVATION_SHAPES,
                action_dim=20,
                max_episode_steps=2,
                device="cpu",
            )

    def test_study_metrics_payload_matches_existing_schema(self):
        payload = study_metrics_payload(
            task_name="block",
            total_timesteps=100,
            checkpoint_steps=[20, 100],
            success_curve=[0.25, 0.5],
            seed=7,
            object_id="obj",
            candidate_id="candidate",
            physics_mode="rigid",
        )
        self.assertEqual(payload["tasks"], ["block"])
        self.assertEqual(payload["checkpoints"], [0.2, 1.0])
        self.assertEqual(payload["success"], {"block": [0.25, 0.5]})
        self.assertEqual(payload["final_success"], {"block": 0.5})
        self.assertEqual(payload["backend"], "mujoco_warp")

    def test_auto_num_envs_uses_only_complete_loop_report_for_matching_xml(self):
        with tempfile.TemporaryDirectory(prefix="shadowhand-auto-env-test.") as root:
            root_path = Path(root)
            xml = root_path / "model.xml"
            xml.write_text("<mujoco/>", encoding="utf-8")
            report = root_path / "benchmark.json"
            report.write_text(
                "{\n"
                f'  "xml": "{xml}",\n'
                '  "recommendation_basis": "maximum measured complete-loop transitions_per_second",\n'
                '  "update_ratio_basis": "one gradient update per 6 collected transitions (SB3 six-env reference)",\n'
                '  "recommended_num_envs": 64,\n'
                '  "results": [{"ok": true, "worlds": 16}, '
                '{"ok": true, "worlds": 64, "capacity_high_water": '
                '{"batch_global_active_contacts": 240, '
                '"max_constraints_per_world": 113, "overflow_flags": 0, '
                '"contacts_per_world": 128, "constraints_per_world": 256}}]\n'
                "}\n",
                encoding="utf-8",
            )
            self.assertEqual(load_num_envs_recommendation(report, xml_path=xml), 64)
            self.assertEqual(
                load_complete_loop_recommendation(report, xml_path=xml),
                AutoTuneRecommendation(64, 128, 256),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_num_envs_recommendation(report, xml_path=root_path / "other.xml")

            unsafe = json.loads(report.read_text(encoding="utf-8"))
            unsafe["results"][1]["capacity_high_water"]["overflow_flags"] = 1
            report.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "capacity overflow"):
                load_complete_loop_recommendation(report, xml_path=xml)

    def test_reference_gradient_steps_preserves_six_env_update_ratio(self):
        self.assertEqual(reference_gradient_steps(6), 1)
        self.assertEqual(reference_gradient_steps(64), 11)
        self.assertEqual(reference_gradient_steps(1024), 171)
        with self.assertRaisesRegex(ValueError, "positive"):
            reference_gradient_steps(0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import numpy as np
import torch
from gymnasium import spaces
from gymnasium_robotics.utils import rotations
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from shadowhand_gpu.rl.normalization import (
    CudaVecNormalize,
    TorchRunningMeanStd,
    flatten_observations,
)
from shadowhand_gpu.rl.replay import (
    CudaHERReplayBuffer,
    plan_replay_memory,
    shadowhand_sparse_reward,
)


OBSERVATION_SHAPES = {
    "achieved_goal": (7,),
    "desired_goal": (7,),
    "observation": (3,),
}


class NormalizationParityTests(unittest.TestCase):
    def test_running_moments_match_sb3(self):
        rng = np.random.default_rng(31)
        reference = RunningMeanStd(shape=(5,))
        running = TorchRunningMeanStd((5,), device="cpu")
        for batch_size in (4, 7, 3):
            values = rng.normal(size=(batch_size, 5)).astype(np.float32)
            reference.update(values)
            running.update(torch.from_numpy(values))
            np.testing.assert_allclose(running.mean.numpy(), reference.mean, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(running.var.numpy(), reference.var, rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(float(running.count), float(reference.count), places=14)

    def test_dict_observation_and_reward_updates_match_vecnormalize_equations(self):
        rng = np.random.default_rng(32)
        num_envs = 3
        normalizer = CudaVecNormalize(
            OBSERVATION_SHAPES,
            num_envs=num_envs,
            gamma=0.95,
            device="cpu",
        )
        reference_obs = {key: RunningMeanStd(shape=shape) for key, shape in OBSERVATION_SHAPES.items()}
        reference_return = RunningMeanStd(shape=())
        returns = np.zeros(num_envs)

        initial = {
            key: rng.normal(size=(num_envs, *shape)).astype(np.float32)
            for key, shape in OBSERVATION_SHAPES.items()
        }
        normalized_initial = normalizer.reset(
            {key: torch.from_numpy(value) for key, value in initial.items()}
        )
        for key in reference_obs:
            reference_obs[key].update(initial[key])
            expected = np.clip(
                (initial[key] - reference_obs[key].mean) / np.sqrt(reference_obs[key].var + 1e-8),
                -10.0,
                10.0,
            ).astype(np.float32)
            np.testing.assert_allclose(normalized_initial[key].numpy(), expected, rtol=1e-6, atol=1e-7)

        for step in range(5):
            observations = {
                key: rng.normal(size=(num_envs, *shape)).astype(np.float32)
                for key, shape in OBSERVATION_SHAPES.items()
            }
            rewards = rng.normal(size=num_envs).astype(np.float32)
            dones = np.array([step == 2, False, step == 4])
            normalized_obs, normalized_reward = normalizer.step(
                {key: torch.from_numpy(value) for key, value in observations.items()},
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
            )
            for key in reference_obs:
                reference_obs[key].update(observations[key])
                expected = np.clip(
                    (observations[key] - reference_obs[key].mean)
                    / np.sqrt(reference_obs[key].var + 1e-8),
                    -10.0,
                    10.0,
                ).astype(np.float32)
                np.testing.assert_allclose(normalized_obs[key].numpy(), expected, rtol=1e-6, atol=1e-7)
            returns = returns * 0.95 + rewards
            reference_return.update(returns)
            expected_reward = np.clip(
                rewards / np.sqrt(reference_return.var + 1e-8),
                -10.0,
                10.0,
            ).astype(np.float32)
            np.testing.assert_allclose(normalized_reward.numpy(), expected_reward, rtol=1e-6, atol=1e-7)
            returns[dones] = 0.0
            np.testing.assert_allclose(normalizer.returns.numpy(), returns, rtol=1e-14, atol=1e-14)

    def test_flatten_order_matches_gym_dict_and_sb3_combined_extractor(self):
        observations = {
            "observation": torch.tensor([[30.0, 31.0]]),
            "achieved_goal": torch.tensor([[10.0]]),
            "desired_goal": torch.tensor([[20.0]]),
        }
        flattened = flatten_observations(observations)
        torch.testing.assert_close(flattened, torch.tensor([[10.0, 20.0, 30.0, 31.0]]))


def _step_values(step, num_envs=2):
    env = torch.arange(num_envs, dtype=torch.float32)
    achieved = torch.zeros(num_envs, 7)
    achieved[:, 0] = step + env * 0.01
    achieved[:, 3] = 1.0
    next_achieved = achieved.clone()
    next_achieved[:, 0] += 0.1
    desired = torch.zeros(num_envs, 7)
    desired[:, 0] = 9.0
    desired[:, 3] = 1.0
    observations = {
        "achieved_goal": achieved,
        "desired_goal": desired,
        "observation": torch.stack((env, env + step, env - step), dim=1),
    }
    next_observations = {
        "achieved_goal": next_achieved,
        "desired_goal": desired.clone(),
        "observation": observations["observation"] + 0.5,
    }
    actions = torch.stack((env + step, env - step), dim=1) / 10.0
    rewards = torch.tensor([-1.0, float(step)], dtype=torch.float32)
    return observations, next_observations, actions, rewards


class _RewardEnv:
    def env_method(self, method_name, achieved_goal, desired_goal, infos, indices):
        assert method_name == "compute_reward"
        achieved = torch.from_numpy(np.asarray(achieved_goal))
        desired = torch.from_numpy(np.asarray(desired_goal))
        return [shadowhand_sparse_reward(achieved, desired).numpy()]


def _sb3_buffer(total_capacity=12, num_envs=2):
    observation_space = spaces.Dict(
        {
            key: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
            for key, shape in OBSERVATION_SHAPES.items()
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return HerReplayBuffer(
        total_capacity,
        observation_space,
        action_space,
        env=_RewardEnv(),
        device="cpu",
        n_envs=num_envs,
        n_sampled_goal=4,
    )


class ReplayParityTests(unittest.TestCase):
    def _buffer(self, total_capacity=12, num_envs=2, max_episode_steps=3):
        return CudaHERReplayBuffer(
            requested_capacity=total_capacity,
            num_envs=num_envs,
            observation_shapes=OBSERVATION_SHAPES,
            action_dim=2,
            max_episode_steps=max_episode_steps,
            device="cpu",
            memory_budget_bytes=10_000_000,
        )

    def test_memory_plan_fails_explicit_capacity_and_records_auto_capacity(self):
        project_shapes = {
            "achieved_goal": (7,),
            "desired_goal": (7,),
            "observation": (562,),
        }
        with self.assertRaisesRegex(MemoryError, "does not fit"):
            plan_replay_memory(
                requested_capacity=1_000_000,
                num_envs=1,
                observation_shapes=project_shapes,
                action_dim=20,
                max_episode_steps=100,
                device="cpu",
                memory_budget_bytes=3_680_000_000,
            )
        with self.assertRaisesRegex(MemoryError, "complete vectorized episode"):
            plan_replay_memory(
                requested_capacity=1_000_000,
                num_envs=4,
                observation_shapes=project_shapes,
                action_dim=20,
                max_episode_steps=100,
                device="cpu",
                memory_budget_bytes=1_000_000,
                auto_capacity=True,
            )
        plan = plan_replay_memory(
            requested_capacity=1_000_000,
            num_envs=4,
            observation_shapes=project_shapes,
            action_dim=20,
            max_episode_steps=100,
            device="cpu",
            memory_budget_bytes=2_000_000,
            auto_capacity=True,
        )
        self.assertLess(plan.effective_capacity, plan.requested_capacity)
        self.assertEqual(plan.effective_capacity % 4, 0)
        self.assertGreaterEqual(plan.rows, 100)
        self.assertLessEqual(plan.storage_bytes, 2_000_000)

    def test_sparse_reward_matches_gymnasium_robotics_formula(self):
        rng = np.random.default_rng(33)
        achieved = rng.normal(size=(32, 7)).astype(np.float32)
        desired = rng.normal(size=(32, 7)).astype(np.float32)
        achieved[:, 3:] /= np.linalg.norm(achieved[:, 3:], axis=1, keepdims=True)
        desired[:, 3:] /= np.linalg.norm(desired[:, 3:], axis=1, keepdims=True)
        delta_position = achieved[:, :3] - desired[:, :3]
        position_distance = np.linalg.norm(delta_position, axis=-1)
        difference = rotations.quat_mul(achieved[:, 3:], rotations.quat_conjugate(desired[:, 3:]))
        rotation_distance = 2 * np.arccos(np.clip(difference[:, 0], -1.0, 1.0))
        expected = (
            ((position_distance < 0.01) & (rotation_distance < 0.1)).astype(np.float32) - 1.0
        )
        actual = shadowhand_sparse_reward(torch.from_numpy(achieved), torch.from_numpy(desired))
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_ignore_z_reward_preserves_online_and_legacy_her_batch_semantics(self):
        achieved = np.zeros((5, 7), dtype=np.float32)
        desired = np.zeros((5, 7), dtype=np.float32)
        desired[:, 3:] = rotations.euler2quat(np.zeros((5, 3)))
        achieved_euler = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.2, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
        )
        achieved[:, 3:] = rotations.euler2quat(achieved_euler)
        achieved_tensor = torch.from_numpy(achieved)
        desired_tensor = torch.from_numpy(desired)

        online = shadowhand_sparse_reward(
            achieved_tensor,
            desired_tensor,
            ignore_z_rotation=True,
        )
        legacy_her = shadowhand_sparse_reward(
            achieved_tensor,
            desired_tensor,
            ignore_z_rotation=True,
            legacy_ignore_z_batch_semantics=True,
        )
        np.testing.assert_array_equal(online.numpy(), np.array([0.0, 0.0, -1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(legacy_her.numpy(), np.array([-1.0, -1.0, 0.0, -1.0, -1.0]))

    def test_episode_metadata_and_fixed_her_batch_match_sb3(self):
        buffer = self._buffer()
        reference = _sb3_buffer()
        for step in range(3):
            observations, next_observations, actions, rewards = _step_values(step)
            dones = torch.tensor([step == 2, step == 2])
            timeouts = torch.tensor([False, step == 2])
            buffer.add(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                timeouts=timeouts,
            )
            reference.add(
                {key: value.numpy() for key, value in observations.items()},
                {key: value.numpy() for key, value in next_observations.items()},
                actions.numpy(),
                rewards.numpy(),
                dones.numpy(),
                [
                    {"TimeLimit.truncated": bool(timeouts[0])},
                    {"TimeLimit.truncated": bool(timeouts[1])},
                ],
            )

        np.testing.assert_array_equal(buffer.episode_start.numpy(), reference.ep_start)
        np.testing.assert_array_equal(buffer.episode_length.numpy(), reference.ep_length)
        rows = torch.tensor([0, 1, 2, 0, 2])
        envs = torch.tensor([0, 0, 1, 1, 1])
        future_rows = torch.tensor([1, 2, 2, 0])
        sample = buffer.sample_from_indices(
            rows,
            envs,
            virtual_count=4,
            future_rows=future_rows,
        )

        virtual_rows = rows[:4].numpy()
        virtual_envs = envs[:4].numpy()
        real_rows = rows[4:].numpy()
        real_envs = envs[4:].numpy()
        reference._sample_goals = lambda batch_rows, batch_envs: reference.next_observations[
            "achieved_goal"
        ][future_rows.numpy(), batch_envs]
        real = reference._get_real_samples(real_rows, real_envs)
        virtual = reference._get_virtual_samples(virtual_rows, virtual_envs)
        for key in OBSERVATION_SHAPES:
            expected_obs = torch.cat((real.observations[key], virtual.observations[key]))
            expected_next = torch.cat((real.next_observations[key], virtual.next_observations[key]))
            torch.testing.assert_close(sample.observations[key], expected_obs, rtol=0.0, atol=0.0)
            torch.testing.assert_close(sample.next_observations[key], expected_next, rtol=0.0, atol=0.0)
        torch.testing.assert_close(sample.actions, torch.cat((real.actions, virtual.actions)))
        torch.testing.assert_close(sample.dones, torch.cat((real.dones, virtual.dones)))
        torch.testing.assert_close(sample.rewards, torch.cat((real.rewards, virtual.rewards)))
        self.assertFalse(bool(sample.dones[0]))  # The real transition is a timeout.
        self.assertEqual(sample.virtual.tolist(), [False, True, True, True, True])

    def test_overwriting_one_member_invalidates_the_complete_old_episode(self):
        buffer = self._buffer(total_capacity=8, max_episode_steps=2)
        for step in range(4):
            observations, next_observations, actions, rewards = _step_values(step)
            dones = torch.tensor([step % 2 == 1, step % 2 == 1])
            buffer.add(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                timeouts=torch.zeros(2, dtype=torch.bool),
            )
        self.assertTrue(buffer.full)
        observations, next_observations, actions, rewards = _step_values(4)
        buffer.add(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=torch.zeros(2, dtype=torch.bool),
            timeouts=torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(buffer.episode_length[:2].count_nonzero().item(), 0)
        self.assertEqual(buffer.episode_length[2:].tolist(), [[2, 2], [2, 2]])

    @unittest.skipUnless(
        os.environ.get("SHADOWHAND_RUN_CUDA_RL_TESTS") == "1" and torch.cuda.is_available(),
        "set SHADOWHAND_RUN_CUDA_RL_TESTS=1 on a CUDA host",
    )
    def test_normalized_her_sampling_remains_cuda_resident(self):
        buffer = CudaHERReplayBuffer(
            requested_capacity=24,
            num_envs=2,
            observation_shapes=OBSERVATION_SHAPES,
            action_dim=2,
            max_episode_steps=3,
            device="cuda",
            memory_budget_bytes=10_000_000,
        )
        normalizer = CudaVecNormalize(
            OBSERVATION_SHAPES,
            num_envs=2,
            device="cuda",
        )
        for step in range(3):
            observations, next_observations, actions, rewards = _step_values(step)
            cuda_observations = {key: value.cuda() for key, value in observations.items()}
            cuda_next = {key: value.cuda() for key, value in next_observations.items()}
            dones = torch.tensor([step == 2, step == 2], device="cuda")
            if step == 0:
                normalizer.reset(cuda_observations)
            normalizer.step(cuda_next, rewards.cuda(), dones)
            buffer.add(
                observations=cuda_observations,
                next_observations=cuda_next,
                actions=actions.cuda(),
                rewards=rewards.cuda(),
                dones=dones,
                timeouts=torch.zeros(2, dtype=torch.bool, device="cuda"),
            )
        sample = buffer.sample(16, normalizer=normalizer)
        tqc_batch = sample.to_tqc_batch()
        torch.cuda.synchronize()
        self.assertTrue(tqc_batch.observations.is_cuda)
        self.assertTrue(tqc_batch.next_observations.is_cuda)
        self.assertTrue(tqc_batch.rewards.is_cuda)
        self.assertEqual(tuple(tqc_batch.observations.shape), (16, 17))


if __name__ == "__main__":
    unittest.main()

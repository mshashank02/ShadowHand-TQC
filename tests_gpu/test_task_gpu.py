import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


RUN_MJW = (
    os.environ.get("SHADOWHAND_RUN_MJW_TESTS") == "1"
    and importlib.util.find_spec("mujoco_warp") is not None
)


def _generate_native_block(output_root):
    from pipeline_generate import build_candidate_standalone

    repository = Path(__file__).resolve().parents[1]
    with contextlib.redirect_stdout(io.StringIO()):
        return build_candidate_standalone(
            task="block",
            Ntotal=16,
            Rppx=0.4,
            Rpt=0.3,
            Ap=6557,
            Apx=26885,
            At=7193,
            Ap1=5557,
            Ap2=1000,
            base_xml=str(repository / "assets" / "hand_base.xml"),
            template_xml=str(repository / "assets" / "manipulate_block_touch_sensors.xml"),
            out_root=output_root,
            force=True,
        )


def _generate_custom_rigid_mesh(output_root, sensors=16):
    from pipeline_generate import build_candidate_standalone

    repository = Path(__file__).resolve().parents[1]
    with contextlib.redirect_stdout(io.StringIO()):
        return build_candidate_standalone(
            task="custom_rigid_mesh",
            Ntotal=sensors,
            Rppx=0.4,
            Rpt=0.3,
            Ap=6557,
            Apx=26885,
            At=7193,
            Ap1=5557,
            Ap2=1000,
            base_xml=str(repository / "assets" / "hand_base.xml"),
            template_xml=str(repository / "assets" / "manipulate_block_touch_sensors.xml"),
            out_root=output_root,
            force=True,
            custom_msh=str(
                repository
                / "study_objects/sphere_study_v1/obj_size-large_ar-high_macro-high_rough-high.msh"
            ),
            deformable_object=False,
            flex_scale="0.03125 0.03125 0.03125",
            flex_radius="0.00125",
            object_pos="1 0.87 0.46",
            object_mass=0.976562,
            object_inertia="0.00305176 0.00305176 0.00305176",
            rigid_mesh_cache_dir=str(Path(output_root) / "rigid_mesh_cache"),
        )


@unittest.skipUnless(RUN_MJW, "set SHADOWHAND_RUN_MJW_TESTS=1 in the isolated GPU environment")
class ShadowHandWarpTaskIntegrationTests(unittest.TestCase):
    def test_real_custom_mesh_n500_n1000_observation_replay_and_capacity_contract(self):
        import torch

        from shadowhand_gpu.rl.replay import CudaHERReplayBuffer
        from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        with tempfile.TemporaryDirectory(prefix="shadowhand-custom-contract-test.") as output_root:
            for sensors in (500, 1000):
                paths = _generate_custom_rigid_mesh(output_root, sensors=sensors)
                backend = MujocoWarpBackend(
                    paths["env"],
                    worlds=2,
                    contacts_per_world=128,
                    constraints_per_world=256,
                )
                task = ShadowHandWarpTask(
                    backend,
                    config=ShadowHandTaskConfig(max_episode_steps=2),
                    seed=123,
                )
                observations = task.reset()
                result = task.step(torch.zeros((2, 20), device="cuda"))
                backend.synchronize()
                self.assertEqual(len(backend.sensor_layout.touch_data_indices), sensors)
                self.assertEqual(task.observation_width, sensors + 62)
                self.assertEqual(tuple(observations["observation"].shape), (2, sensors + 62))
                self.assertEqual(tuple(result.observations["observation"].shape), (2, sensors + 62))
                replay = CudaHERReplayBuffer(
                    requested_capacity=400,
                    num_envs=2,
                    observation_shapes={
                        "achieved_goal": (7,),
                        "desired_goal": (7,),
                        "observation": (sensors + 62,),
                    },
                    action_dim=20,
                    max_episode_steps=2,
                    n_sampled_goal=4,
                    reward_function=task.compute_her_rewards,
                    device="cuda",
                )
                self.assertEqual(replay.plan.effective_capacity, 400)
                self.assertEqual(int(backend.overflow_flags.max().cpu()), 0)
                self.assertLessEqual(int(backend.active_contact_counts.max().cpu()), 2 * 128)
                self.assertLessEqual(int(backend.constraint_counts.max().cpu()), 256)

    def test_custom_rigid_mesh_policy_step_reward_and_success_parity(self):
        import mujoco
        import torch

        from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        with tempfile.TemporaryDirectory(prefix="shadowhand-custom-mesh-task-test.") as output_root:
            paths = _generate_custom_rigid_mesh(output_root)
            backend = MujocoWarpBackend(
                paths["env"],
                worlds=1,
                contacts_per_world=128,
                constraints_per_world=256,
            )
            self.assertEqual(backend.model_report.object_collision_representation, "rigid_mesh_geom")
            self.assertEqual(backend.model_report.nflex, 0)
            task = ShadowHandWarpTask(
                backend,
                config=ShadowHandTaskConfig(max_episode_steps=2),
                seed=123,
            )
            initial = task.reset()
            backend.synchronize()
            self.assertEqual(tuple(initial["observation"].shape), (1, 78))

            actions = torch.linspace(-0.5, 0.5, 20, device="cuda").repeat(1, 1)
            task.apply_actions(actions)
            backend.synchronize()
            cpu_data = mujoco.MjData(backend.model)
            cpu_data.qpos[:] = backend.qpos[0].cpu().numpy()
            cpu_data.qvel[:] = backend.qvel[0].cpu().numpy()
            cpu_data.ctrl[:] = backend.ctrl[0].cpu().numpy()
            cpu_data.qacc_warmstart[:] = backend.qacc_warmstart[0].cpu().numpy()
            cpu_data.time = float(backend.time[0].cpu())
            mujoco.mj_step(backend.model, cpu_data, nstep=20)

            result = task.step(actions)
            backend.synchronize()
            np.testing.assert_allclose(backend.qpos[0].cpu().numpy(), cpu_data.qpos, atol=3e-4, rtol=0)
            np.testing.assert_allclose(backend.qvel[0].cpu().numpy(), cpu_data.qvel, atol=2e-2, rtol=0)
            touch_indices = np.asarray(backend.sensor_layout.touch_data_indices)
            np.testing.assert_allclose(
                backend.touch[0].cpu().numpy(),
                cpu_data.sensordata[touch_indices],
                atol=0.2,
                rtol=0,
            )
            start = task.layout.object_qpos_start
            cpu_achieved = torch.as_tensor(
                cpu_data.qpos[start : start + 7],
                dtype=backend.qpos.dtype,
                device="cuda",
            ).unsqueeze(0)
            expected_reward = task.compute_rewards(cpu_achieved, task.goals)
            self.assertTrue(torch.equal(result.rewards, expected_reward))
            self.assertTrue(torch.equal(result.success, expected_reward + 1.0))

    def test_native_block_task_parity_timeout_and_masked_reset(self):
        import mujoco
        import torch

        from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        with tempfile.TemporaryDirectory(prefix="shadowhand-native-task-test.") as output_root:
            paths = _generate_native_block(output_root)
            backend = MujocoWarpBackend(
                paths["env"],
                worlds=2,
                contacts_per_world=2048,
                constraints_per_world=2048,
            )
            task = ShadowHandWarpTask(
                backend,
                config=ShadowHandTaskConfig(max_episode_steps=2),
                seed=123,
            )
            initial = task.reset()
            backend.synchronize()
            self.assertEqual(tuple(initial["observation"].shape), (2, 78))
            self.assertEqual(tuple(initial["achieved_goal"].shape), (2, 7))
            self.assertTrue(initial["observation"].is_cuda)
            self.assertTrue(torch.all(initial["achieved_goal"][:, 2] > 0.04))
            self.assertEqual(task.episode_ids.tolist(), [0, 0])
            position_offset = initial["desired_goal"][:, :3] - initial["achieved_goal"][:, :3]
            self.assertTrue(torch.all((position_offset[:, 0] >= -0.04) & (position_offset[:, 0] <= 0.04)))
            self.assertTrue(torch.all((position_offset[:, 1] >= -0.06) & (position_offset[:, 1] <= 0.02)))
            self.assertTrue(torch.all((position_offset[:, 2] >= 0.0) & (position_offset[:, 2] <= 0.06)))

            actions = torch.linspace(-0.5, 0.5, 20, device="cuda").repeat(2, 1)
            task.apply_actions(actions)
            backend.synchronize()
            cpu_data = mujoco.MjData(backend.model)
            cpu_data.qpos[:] = backend.qpos[0].cpu().numpy()
            cpu_data.qvel[:] = backend.qvel[0].cpu().numpy()
            cpu_data.ctrl[:] = backend.ctrl[0].cpu().numpy()
            cpu_data.qacc_warmstart[:] = backend.qacc_warmstart[0].cpu().numpy()
            cpu_data.time = float(backend.time[0].cpu())
            mujoco.mj_step(backend.model, cpu_data, nstep=20)

            first_step = task.step(actions)
            backend.synchronize()
            np.testing.assert_allclose(
                backend.qpos[0].cpu().numpy(),
                cpu_data.qpos,
                rtol=0.0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                backend.qvel[0].cpu().numpy(),
                cpu_data.qvel,
                rtol=0.0,
                atol=3e-5,
            )
            touch_indices = np.asarray(backend.sensor_layout.touch_data_indices)
            np.testing.assert_allclose(
                backend.touch[0].cpu().numpy(),
                cpu_data.sensordata[touch_indices],
                rtol=0.0,
                atol=3e-4,
            )
            self.assertFalse(bool(first_step.truncated.any()))
            self.assertEqual(float(first_step.observations["observation"][0, -1]), 0.5)

            second_step = task.step(torch.zeros(2, 20, device="cuda"))
            backend.synchronize()
            self.assertTrue(bool(second_step.truncated.all()))
            self.assertFalse(bool(second_step.terminated.any()))

            protected = {
                "qpos": backend.qpos[1].clone(),
                "qvel": backend.qvel[1].clone(),
                "sensordata": backend.sensordata[1].clone(),
                "time": backend.time[1].clone(),
                "goal": task.goals[1].clone(),
                "episode_steps": task.episode_steps[1].clone(),
            }
            reset_observations = task.reset(torch.tensor([True, False], device="cuda"))
            backend.synchronize()
            self.assertTrue(torch.equal(backend.qpos[1], protected["qpos"]))
            self.assertTrue(torch.equal(backend.qvel[1], protected["qvel"]))
            self.assertTrue(torch.equal(backend.sensordata[1], protected["sensordata"]))
            self.assertTrue(torch.equal(backend.time[1], protected["time"]))
            self.assertTrue(torch.equal(task.goals[1], protected["goal"]))
            self.assertTrue(torch.equal(task.episode_steps[1], protected["episode_steps"]))
            self.assertEqual(int(task.episode_steps[0]), 0)
            self.assertEqual(task.episode_ids.tolist(), [1, 0])
            self.assertEqual(float(reset_observations["observation"][0, -1]), 1.0)

            checkpoint = {
                key: value.clone() for key, value in task.checkpoint().items()
            }
            task.step(torch.ones(2, 20, device="cuda") * 0.1)
            task.load_checkpoint(checkpoint)
            backend.synchronize()
            for key, expected in checkpoint.items():
                if key == "rng_state":
                    actual = task.rng.state
                elif hasattr(task, key):
                    actual = getattr(task, key)
                else:
                    actual = getattr(backend, key)
                self.assertTrue(torch.equal(actual, expected), key)

            from shadowhand_gpu.rl.normalization import CudaVecNormalize
            from shadowhand_gpu.rl.replay import CudaHERReplayBuffer
            from shadowhand_gpu.rl.tqc import TQCConfig, TQCLearner
            from shadowhand_gpu.trainer import CudaTQCTrainer, TrainerConfig

            observation_shapes = {
                "achieved_goal": (7,),
                "desired_goal": (7,),
                "observation": (task.observation_width,),
            }
            normalizer = CudaVecNormalize(
                observation_shapes,
                num_envs=2,
                device="cuda",
            )
            learner = TQCLearner(
                TQCConfig(
                    observation_dim=task.observation_width + 14,
                    action_dim=20,
                    hidden_dims=(16,),
                    n_quantiles=3,
                    top_quantiles_to_drop_per_critic=1,
                    device="cuda",
                )
            )
            replay = CudaHERReplayBuffer(
                requested_capacity=8,
                num_envs=2,
                observation_shapes=observation_shapes,
                action_dim=20,
                max_episode_steps=2,
                reward_function=task.compute_her_rewards,
                device="cuda",
                memory_budget_bytes=10_000_000,
            )
            trainer = CudaTQCTrainer(
                task=task,
                normalizer=normalizer,
                replay=replay,
                learner=learner,
                config=TrainerConfig(batch_size=4, learning_starts=0),
                seed=123,
            )
            trainer.collect_and_update()
            trained_step = trainer.collect_and_update()
            self.assertIsNotNone(trained_step.update_metrics)
            self.assertTrue(trained_step.actions.is_cuda)
            self.assertTrue(trained_step.update_metrics["actor_loss"].is_cuda)
            evaluation = trainer.evaluate(episodes=2)
            self.assertEqual(evaluation.episodes, 2)
            self.assertEqual(evaluation.timestep, 4)


if __name__ == "__main__":
    unittest.main()

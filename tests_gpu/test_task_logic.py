from pathlib import Path
import unittest

import numpy as np
import torch

from shadowhand_gpu.model_loader import load_project_model
from shadowhand_gpu.sensors import build_sensor_layout
from shadowhand_gpu.task import (
    PerWorldRandom,
    ShadowHandTaskConfig,
    TaskModelLayout,
    build_task_observations,
    condition_actions,
)


N500_XML = Path(
    "generated/smoke_tests/rigid_n0500_a0p3_b0p6_large_high_high_high/"
    "custom_obj_size_large_ar_high_macro_high_rough_high_500_1.071429_0.714286/"
    "manipulate_custom_obj_size_large_ar_high_macro_high_rough_high_touch_sensors_"
    "500_1.071429_0.714286.xml"
)


@unittest.skipUnless(N500_XML.is_file(), "real generated N=500 fixture is absent")
class TaskLogicParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = load_project_model(N500_XML)
        cls.layout = TaskModelLayout.from_model(cls.model)

    def test_compiled_layout_matches_shadowhand_contract(self):
        self.assertEqual(self.layout.robot_qpos_indices, tuple(range(24)))
        self.assertEqual(self.layout.robot_qvel_indices, tuple(range(24)))
        self.assertEqual(self.layout.object_qpos_start, 24)
        self.assertEqual(self.layout.object_qvel_start, 24)
        self.assertEqual(self.layout.target_qpos_start, 31)
        self.assertEqual(len(self.layout.control_centers), 20)

    def test_action_conditioning_and_control_mapping_match_legacy_order(self):
        rng = np.random.default_rng(41)
        actions = rng.normal(size=(4, 20)).astype(np.float32) * 2.0
        previous = rng.normal(size=(4, 20)).astype(np.float32) * 0.2
        actual = condition_actions(
            torch.from_numpy(actions),
            torch.from_numpy(previous),
            action_scale=0.7,
            action_clip=0.6,
            action_smoothing=0.4,
        )
        expected = np.clip(actions, -1.0, 1.0) * 0.7
        expected = np.clip(expected, -0.6, 0.6)
        expected = 0.4 * previous + 0.6 * expected
        expected = np.clip(expected, -0.6, 0.6)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=0.0)

        centers = np.asarray(self.layout.control_centers)
        half_ranges = np.asarray(self.layout.control_half_ranges)
        lows = np.asarray(self.layout.control_lows)
        highs = np.asarray(self.layout.control_highs)
        controls = np.clip(centers + expected * half_ranges, lows, highs)
        self.assertTrue(np.all(controls >= lows))
        self.assertTrue(np.all(controls <= highs))

    def test_fixed_state_observation_matches_gym_robot_joint_order(self):
        import mujoco
        from gymnasium_robotics.utils.mujoco_utils import robot_get_obs
        from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        data.qvel[:] = np.linspace(-0.3, 0.3, self.model.nv)
        mujoco.mj_forward(self.model, data)
        sensor_layout = build_sensor_layout(self.model)
        touch = data.sensordata[np.asarray(sensor_layout.touch_data_indices)]
        desired = np.array([[1.0, 0.8, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        observations = build_task_observations(
            qpos=torch.from_numpy(data.qpos.astype(np.float32)).unsqueeze(0),
            qvel=torch.from_numpy(data.qvel.astype(np.float32)).unsqueeze(0),
            touch=torch.from_numpy(touch.astype(np.float32)).unsqueeze(0),
            desired_goals=torch.from_numpy(desired),
            episode_steps=torch.tensor([25]),
            max_episode_steps=100,
            robot_qpos_indices=torch.tensor(self.layout.robot_qpos_indices),
            robot_qvel_indices=torch.tensor(self.layout.robot_qvel_indices),
            object_qpos_start=self.layout.object_qpos_start,
            object_qvel_start=self.layout.object_qvel_start,
        )
        model_names = MujocoModelNames(self.model)
        robot_qpos, robot_qvel = robot_get_obs(self.model, data, model_names.joint_names)
        expected = np.concatenate(
            (
                robot_qpos,
                robot_qvel,
                data.qvel[self.layout.object_qvel_start : self.layout.object_qvel_start + 6],
                data.qpos[self.layout.object_qpos_start : self.layout.object_qpos_start + 7],
                touch,
                [0.75],
            )
        ).astype(np.float32)
        np.testing.assert_allclose(
            observations["observation"][0].numpy(),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(tuple(observations["observation"].shape), (1, 562))
        np.testing.assert_array_equal(observations["desired_goal"].numpy(), desired)

    def test_per_world_rng_is_reproducible_and_masked_streams_do_not_advance(self):
        first = PerWorldRandom(3, 123, torch.device("cpu"))
        second = PerWorldRandom(3, 123, torch.device("cpu"))
        torch.testing.assert_close(first.uniform(5), second.uniform(5), rtol=0.0, atol=0.0)
        state_before = first.state.clone()
        first.uniform(3, mask=torch.tensor([True, False, True]))
        self.assertEqual(int(first.state[1]), int(state_before[1]))
        self.assertNotEqual(int(first.state[0]), int(state_before[0]))
        self.assertNotEqual(int(first.state[2]), int(state_before[2]))

    def test_ignore_z_mode_is_an_explicit_supported_configuration(self):
        self.assertTrue(ShadowHandTaskConfig(ignore_z_rotation=True).ignore_z_rotation)


if __name__ == "__main__":
    unittest.main()

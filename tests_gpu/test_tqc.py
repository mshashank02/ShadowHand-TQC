import os
import unittest

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import polyak_update as sb3_polyak_update

from sb3_contrib.common.utils import quantile_huber_loss as sb3_quantile_huber_loss
from sb3_contrib.tqc.policies import Actor as SB3Actor
from sb3_contrib.tqc.policies import Critic as SB3Critic
from shadowhand_gpu.rl.tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    TQCBatch,
    TQCConfig,
    TQCLearner,
    build_target_quantiles,
    quantile_huber_loss,
)


def _spaces(observation_dim, action_dim):
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(observation_dim,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )
    return observation_space, action_space


def _sb3_actor(observation_dim, action_dim, hidden_dims):
    observation_space, action_space = _spaces(observation_dim, action_dim)
    return SB3Actor(
        observation_space,
        action_space,
        net_arch=list(hidden_dims),
        features_extractor=FlattenExtractor(observation_space),
        features_dim=observation_dim,
    )


def _sb3_critic(observation_dim, action_dim, hidden_dims, n_critics, n_quantiles):
    observation_space, action_space = _spaces(observation_dim, action_dim)
    return SB3Critic(
        observation_space,
        action_space,
        net_arch=list(hidden_dims),
        features_extractor=FlattenExtractor(observation_space),
        features_dim=observation_dim,
        n_critics=n_critics,
        n_quantiles=n_quantiles,
    )


def _copy_actor_to_sb3(source, target):
    target.latent_pi.load_state_dict(source.latent.state_dict())
    target.mu.load_state_dict(source.mean.state_dict())
    target.log_std.load_state_dict(source.log_std.state_dict())


def _copy_critic_to_sb3(source, target):
    for source_network, target_network in zip(source.q_networks, target.q_networks, strict=True):
        target_network.load_state_dict(source_network.state_dict())


def _assert_parameter_sequences_close(test_case, left, right, *, atol=1e-7):
    left_parameters = list(left.parameters())
    right_parameters = list(right.parameters())
    test_case.assertEqual(len(left_parameters), len(right_parameters))
    for left_parameter, right_parameter in zip(left_parameters, right_parameters, strict=True):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=0.0, atol=atol)


class TQCModuleParityTests(unittest.TestCase):
    def test_actor_forward_sample_log_prob_and_gradients_match_sb3(self):
        torch.manual_seed(10)
        actor = SquashedGaussianActor(5, 2, (11, 7))
        reference = _sb3_actor(5, 2, (11, 7))
        _copy_actor_to_sb3(actor, reference)
        observations = torch.randn(6, 5)

        mean, log_std = actor.distribution_parameters(observations)
        ref_mean, ref_log_std, _ = reference.get_action_dist_params(observations)
        torch.testing.assert_close(mean, ref_mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(log_std, ref_log_std, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actor(observations, deterministic=True),
            reference(observations, deterministic=True),
            rtol=0.0,
            atol=0.0,
        )

        torch.manual_seed(73)
        actions, log_prob = actor.action_log_prob(observations)
        torch.manual_seed(73)
        ref_actions, ref_log_prob = reference.action_log_prob(observations)
        torch.testing.assert_close(actions, ref_actions, rtol=0.0, atol=0.0)
        torch.testing.assert_close(log_prob[:, 0], ref_log_prob, rtol=0.0, atol=0.0)

        (actions.sum() + log_prob.sum()).backward()
        (ref_actions.sum() + ref_log_prob.sum()).backward()
        for own, ref in (
            (actor.latent, reference.latent_pi),
            (actor.mean, reference.mu),
            (actor.log_std, reference.log_std),
        ):
            for own_parameter, ref_parameter in zip(own.parameters(), ref.parameters(), strict=True):
                torch.testing.assert_close(
                    own_parameter.grad,
                    ref_parameter.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_critic_forward_and_gradients_match_sb3(self):
        torch.manual_seed(11)
        critic = QuantileCritic(5, 2, (13, 9), n_critics=2, n_quantiles=5)
        reference = _sb3_critic(5, 2, (13, 9), 2, 5)
        _copy_critic_to_sb3(critic, reference)
        observations = torch.randn(7, 5)
        actions = torch.randn(7, 2).tanh()

        quantiles = critic(observations, actions)
        ref_quantiles = reference(observations, actions)
        torch.testing.assert_close(quantiles, ref_quantiles, rtol=0.0, atol=0.0)
        quantiles.square().mean().backward()
        ref_quantiles.square().mean().backward()
        for own_network, ref_network in zip(
            critic.q_networks,
            reference.q_networks,
            strict=True,
        ):
            for own_parameter, ref_parameter in zip(
                own_network.parameters(),
                ref_network.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    own_parameter.grad,
                    ref_parameter.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_quantile_huber_loss_matches_sb3(self):
        torch.manual_seed(12)
        current = torch.randn(8, 2, 5, requires_grad=True)
        target = torch.randn(8, 1, 6)
        loss = quantile_huber_loss(current, target, sum_over_quantiles=False)
        reference = sb3_quantile_huber_loss(current, target, sum_over_quantiles=False)
        torch.testing.assert_close(loss, reference, rtol=0.0, atol=0.0)

        gradient = torch.autograd.grad(loss, current)[0]
        ref_gradient = torch.autograd.grad(reference, current)[0]
        torch.testing.assert_close(gradient, ref_gradient, rtol=0.0, atol=0.0)

    def test_truncated_entropy_adjusted_bellman_target_matches_sb3_formula(self):
        torch.manual_seed(13)
        next_quantiles = torch.randn(4, 2, 5)
        next_log_prob = torch.randn(4, 1)
        rewards = torch.randn(4, 1)
        dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
        entropy_coefficient = torch.tensor([0.37])
        targets = build_target_quantiles(
            next_quantiles=next_quantiles,
            next_log_prob=next_log_prob,
            rewards=rewards,
            dones=dones,
            discount=0.95,
            entropy_coefficient=entropy_coefficient,
            top_drop_per_critic=2,
        )

        sorted_quantiles = torch.sort(next_quantiles.reshape(4, -1), dim=1).values[:, :6]
        reference = rewards + (1.0 - dones) * 0.95 * (
            sorted_quantiles - entropy_coefficient * next_log_prob
        )
        torch.testing.assert_close(targets, reference.unsqueeze(1), rtol=0.0, atol=0.0)


class TQCLearnerParityTests(unittest.TestCase):
    def test_one_optimizer_step_matches_sb3_update_equations(self):
        config = TQCConfig(
            observation_dim=5,
            action_dim=2,
            hidden_dims=(16, 12),
            n_critics=2,
            n_quantiles=5,
            top_quantiles_to_drop_per_critic=2,
            gamma=0.95,
            tau=0.05,
            learning_rate=1e-3,
            target_entropy=-2.0,
            device="cpu",
        )
        torch.manual_seed(20)
        learner = TQCLearner(config)
        reference_actor = _sb3_actor(5, 2, config.hidden_dims)
        reference_critic = _sb3_critic(5, 2, config.hidden_dims, 2, 5)
        reference_target = _sb3_critic(5, 2, config.hidden_dims, 2, 5)
        _copy_actor_to_sb3(learner.actor, reference_actor)
        _copy_critic_to_sb3(learner.critic, reference_critic)
        _copy_critic_to_sb3(learner.critic_target, reference_target)

        optimizer_kwargs = {"lr": config.learning_rate, "eps": config.adam_epsilon}
        actor_optimizer = torch.optim.Adam(reference_actor.parameters(), **optimizer_kwargs)
        critic_optimizer = torch.optim.Adam(reference_critic.parameters(), **optimizer_kwargs)
        reference_log_entropy = torch.zeros(1, requires_grad=True)
        entropy_optimizer = torch.optim.Adam((reference_log_entropy,), **optimizer_kwargs)

        torch.manual_seed(21)
        batch = TQCBatch(
            observations=torch.randn(8, 5),
            actions=torch.randn(8, 2).tanh(),
            next_observations=torch.randn(8, 5),
            rewards=torch.randn(8, 1),
            dones=torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0]]),
        )

        torch.manual_seed(22)
        metrics = learner.update(batch)

        torch.manual_seed(22)
        actions_pi, log_prob = reference_actor.action_log_prob(batch.observations)
        log_prob = log_prob.reshape(-1, 1)
        entropy_coefficient = reference_log_entropy.detach().exp()
        entropy_loss = -(
            reference_log_entropy * (log_prob + config.resolved_target_entropy).detach()
        ).mean()
        entropy_optimizer.zero_grad()
        entropy_loss.backward()
        entropy_optimizer.step()

        with torch.no_grad():
            next_actions, next_log_prob = reference_actor.action_log_prob(batch.next_observations)
            next_quantiles = reference_target(batch.next_observations, next_actions)
            next_quantiles = torch.sort(next_quantiles.reshape(8, -1), dim=1).values[:, :6]
            target_quantiles = next_quantiles - entropy_coefficient * next_log_prob.reshape(-1, 1)
            target_quantiles = batch.rewards + (1.0 - batch.dones) * config.gamma * target_quantiles
            target_quantiles.unsqueeze_(1)

        current_quantiles = reference_critic(batch.observations, batch.actions)
        critic_loss = sb3_quantile_huber_loss(
            current_quantiles,
            target_quantiles,
            sum_over_quantiles=False,
        )
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        policy_quantiles = reference_critic(batch.observations, actions_pi)
        q_policy = policy_quantiles.mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (entropy_coefficient * log_prob - q_policy).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        sb3_polyak_update(reference_critic.parameters(), reference_target.parameters(), config.tau)

        self.assertAlmostEqual(float(metrics["actor_loss"]), float(actor_loss.detach()), places=7)
        self.assertAlmostEqual(float(metrics["critic_loss"]), float(critic_loss.detach()), places=7)
        self.assertAlmostEqual(
            float(metrics["entropy_coefficient_loss"]),
            float(entropy_loss.detach()),
            places=7,
        )
        _assert_parameter_sequences_close(self, learner.actor, reference_actor)
        _assert_parameter_sequences_close(self, learner.critic, reference_critic)
        _assert_parameter_sequences_close(self, learner.critic_target, reference_target)
        torch.testing.assert_close(
            learner.log_entropy_coefficient,
            reference_log_entropy,
            rtol=0.0,
            atol=1e-7,
        )

    @unittest.skipUnless(
        os.environ.get("SHADOWHAND_RUN_CUDA_RL_TESTS") == "1" and torch.cuda.is_available(),
        "set SHADOWHAND_RUN_CUDA_RL_TESTS=1 on a CUDA host",
    )
    def test_update_remains_cuda_resident(self):
        config = TQCConfig(
            observation_dim=17,
            action_dim=4,
            hidden_dims=(32, 32),
            n_quantiles=7,
            device="cuda",
        )
        learner = TQCLearner(config)
        batch_size = 16
        batch = TQCBatch(
            observations=torch.randn(batch_size, 17, device="cuda"),
            actions=torch.randn(batch_size, 4, device="cuda").tanh(),
            next_observations=torch.randn(batch_size, 17, device="cuda"),
            rewards=torch.randn(batch_size, 1, device="cuda"),
            dones=torch.zeros(batch_size, 1, device="cuda"),
        )
        metrics = learner.update(batch)
        torch.cuda.synchronize()
        self.assertTrue(all(parameter.is_cuda for parameter in learner.actor.parameters()))
        self.assertTrue(metrics["actor_loss"].is_cuda)
        self.assertTrue(metrics["critic_loss"].is_cuda)


if __name__ == "__main__":
    unittest.main()

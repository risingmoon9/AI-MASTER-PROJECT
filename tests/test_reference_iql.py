"""Equation, analytic Adam step, initialization and resume checks for IQL-Ref.

Run: python -m unittest discover -s tests -p test_reference_iql.py -v
These tests do not execute JAX or establish complete benchmark reproduction.
"""
import copy
import math
import unittest

import numpy as np
import torch

from hqa_orl.reference_iql import ReferenceIQLAgent, ReferenceIQLConfig


class ReferenceIQLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def make_agent(self, **kwargs):
        cfg = ReferenceIQLConfig(hidden_dim=8, hidden_depth=1, max_steps=20, **kwargs)
        return ReferenceIQLAgent(3, 2, cfg, torch.device('cpu'))

    def batch(self):
        return {'obs': torch.randn(7, 3), 'next_obs': torch.randn(7, 3),
                'actions': torch.rand(7, 2) * 2 - 1,
                'rewards': torch.randn(7, 1), 'dones': torch.tensor([[0.], [0.], [1.], [0.], [1.], [0.], [0.]]),
                'weights': torch.rand(7, 1), 'support': torch.rand(7, 1),
                'uncertainty': torch.rand(7, 1)}

    def assert_models_equal(self, first, second):
        for name in ('actor', 'q', 'target_q', 'v'):
            for a, b in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_official_kitchen_defaults_initialization_and_eval_dropout(self):
        cfg = ReferenceIQLConfig()
        self.assertEqual((cfg.expectile, cfg.temperature, cfg.dropout_rate), (.7, .5, .1))
        agent = self.make_agent()
        for network in (agent.actor, agent.q, agent.v):
            for layer in network.modules():
                if isinstance(layer, torch.nn.Linear):
                    w = layer.weight.detach()
                    gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
                    torch.testing.assert_close(gram, 2 * torch.eye(gram.shape[0]), atol=1e-6, rtol=1e-6)
                    self.assertEqual(torch.count_nonzero(layer.bias).item(), 0)
        self.assertEqual(agent.actor.log_stds.shape, (2,))
        self.assertEqual(torch.count_nonzero(agent.actor.log_stds).item(), 0)
        obs = torch.randn(100, 3)
        rng = torch.get_rng_state().clone()
        a = agent.actor.act(obs)
        torch.testing.assert_close(a, agent.actor.act(obs), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(torch.all(a.abs() <= 1))
        a1, _ = agent.actor(obs, training=True)
        a2, _ = agent.actor(obs, training=True)
        self.assertFalse(torch.equal(a1, a2))

    def test_normal_density_has_no_tanh_jacobian_and_global_std(self):
        agent = self.make_agent(dropout_rate=0.0)
        with torch.no_grad():
            for p in agent.actor.net.parameters():
                p.zero_()
            agent.actor.net.layers[-1].bias.copy_(torch.tensor([.3, -.2]))
            agent.actor.log_stds.copy_(torch.tensor([-10., 5.]))
        obs = torch.randn(3, 3)
        actions = torch.tensor([[1., -1.], [0., 0.], [.5, -.5]])
        mean = np.tanh(np.array([.3, -.2]))
        log_std = np.array([-5., 2.])
        expected = np.sum(-.5 * ((actions.numpy() - mean) * np.exp(-log_std)) ** 2
                          - log_std - .5 * np.log(2 * np.pi), axis=1, keepdims=True)
        np.testing.assert_allclose(agent.actor.log_prob(obs, actions).detach().numpy(), expected,
                                   rtol=2e-6, atol=2e-5)

    def test_full_one_step_against_closed_form_gradients_and_adam(self):
        # Linear, constant initial networks allow an independent analytic oracle
        # for every nonzero gradient and optimizer/target update.
        cfg = ReferenceIQLConfig(hidden_depth=0, hidden_dim=2, dropout_rate=0,
                                 max_steps=20, actor_lr=.013, value_lr=.021,
                                 critic_lr=.017, tau=.2)
        agent = ReferenceIQLAgent(1, 2, cfg, torch.device('cpu'))
        for net in (agent.actor, agent.q, agent.target_q, agent.v):
            net.double()
            with torch.no_grad():
                for p in net.parameters():
                    p.zero_()
        z = np.array([.2, -.4])
        log_std = np.log(np.array([.7, 1.2]))
        with torch.no_grad():
            agent.actor.net.layers[0].bias.copy_(torch.from_numpy(z))
            agent.actor.log_stds.copy_(torch.from_numpy(log_std))
            agent.v.layers[0].bias.fill_(1.)
            for net in (agent.q, agent.target_q):
                net.q1.layers[0].bias.fill_(2.)
                net.q2.layers[0].bias.fill_(3.)
        actions = np.array([[-.3, .2], [.8, -.7]])
        rewards, dones = np.array([1., 2.]), np.array([0., 1.])
        batch = {'obs': torch.zeros(2, 1, dtype=torch.float64),
                 'next_obs': torch.zeros(2, 1, dtype=torch.float64),
                 'actions': torch.from_numpy(actions),
                 'rewards': torch.from_numpy(rewards), 'dones': torch.from_numpy(dones)}
        adam_first = lambda p, grad, lr: p - lr * grad / (np.abs(grad) + 1e-8)
        v_new = adam_first(1., -2 * cfg.expectile, cfg.value_lr)
        aw = math.exp((2. - v_new) * cfg.temperature)
        mean, std = np.tanh(z), np.exp(log_std)
        log_probs = np.sum(-.5 * ((actions - mean) / std) ** 2 - log_std
                           - .5 * np.log(2 * np.pi), axis=1)
        grad_mean = aw * np.mean((mean - actions) / std ** 2, axis=0)
        grad_z = grad_mean * (1 - mean ** 2)
        grad_std = aw * np.mean(1 - ((actions - mean) / std) ** 2, axis=0)
        expected_z = adam_first(z, grad_z, cfg.actor_lr)
        expected_std = adam_first(log_std, grad_std, cfg.actor_lr)
        q_target = rewards + cfg.discount * (1 - dones) * v_new
        expected_q_loss = np.mean((2 - q_target) ** 2 + (3 - q_target) ** 2)
        metrics = agent.update(batch)
        self.assertAlmostEqual(metrics['loss/value'], cfg.expectile, places=12)
        self.assertAlmostEqual(metrics['loss/actor'], -aw * log_probs.mean(), places=12)
        self.assertAlmostEqual(metrics['loss/q'], expected_q_loss, places=12)
        np.testing.assert_allclose(agent.actor.net.layers[0].bias.detach(), expected_z, atol=1e-14)
        np.testing.assert_allclose(agent.actor.log_stds.detach(), expected_std, atol=1e-14)
        np.testing.assert_allclose(agent.v.layers[0].bias.detach(), [v_new], atol=1e-14)
        x = np.concatenate((np.zeros((2, 1)), actions), axis=1)
        for name, initial in [('q1', 2.), ('q2', 3.)]:
            residual = initial - q_target
            grad_bias = np.mean(2 * residual)
            grad_weight = np.mean(2 * residual[:, None] * x, axis=0)[None, :]
            expected_w = adam_first(np.zeros((1, 3)), grad_weight, cfg.critic_lr)
            expected_b = adam_first(initial, grad_bias, cfg.critic_lr)
            layer = getattr(agent.q, name).layers[0]
            target = getattr(agent.target_q, name).layers[0]
            np.testing.assert_allclose(layer.weight.detach(), expected_w, atol=1e-14)
            np.testing.assert_allclose(layer.bias.detach(), [expected_b], atol=1e-14)
            np.testing.assert_allclose(target.weight.detach(), cfg.tau * expected_w, atol=1e-14)
            np.testing.assert_allclose(target.bias.detach(), [(1-cfg.tau)*initial + cfg.tau*expected_b], atol=1e-14)

    def test_cap_and_ignore_quality_and_logging_does_not_change_updates(self):
        agent = self.make_agent(dropout_rate=0)
        other = self.make_agent(dropout_rate=0)
        with torch.no_grad():
            for p in agent.target_q.parameters():
                p.zero_()
            agent.target_q.q1.layers[-1].bias.fill_(1000.)
            agent.target_q.q2.layers[-1].bias.fill_(1000.)
        other.load_state_dict(copy.deepcopy(agent.state_dict()))
        batch = self.batch()
        batch2 = copy.deepcopy(batch)
        for key in ('weights', 'support', 'uncertainty'):
            batch2[key].fill_(float('nan'))
        metrics = agent.update(batch)
        self.assertEqual(metrics['stats/awr_mean'], 100.)
        self.assertTrue(np.isfinite(metrics['loss/actor']))
        self.assertEqual(other.update(batch2, collect_metrics=False), {})
        self.assert_models_equal(agent, other)

    def test_cosine_schedule_and_exact_resume_with_dropout_and_optimizer_moments(self):
        agent = self.make_agent()
        batch = self.batch()
        self.assertEqual(agent.actor_learning_rate(), agent.cfg.actor_lr)
        for _ in range(10):
            agent.update(batch, collect_metrics=False)
        self.assertAlmostEqual(agent.actor_learning_rate(), agent.cfg.actor_lr / 2)
        state = copy.deepcopy(agent.state_dict())
        rng = torch.get_rng_state().clone()
        result = agent.update(batch)
        restored = self.make_agent()
        restored.load_state_dict(state)
        torch.set_rng_state(rng)
        result2 = restored.update(batch)
        self.assertEqual(result, result2)
        self.assertEqual(agent.total_updates, restored.total_updates)
        self.assert_models_equal(agent, restored)
        agent.total_updates = 20
        self.assertEqual(agent.actor_learning_rate(), 0)
        agent.total_updates = 21
        self.assertEqual(agent.actor_learning_rate(), 0)
        incompatible = self.make_agent(temperature=3.0)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            incompatible.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()

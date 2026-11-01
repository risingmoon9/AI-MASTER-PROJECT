"""Small numerical checks for the objectives and observation-data contract."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hqa_orl.bc import BCAgent, BCConfig
from hqa_orl.iql import IQLAgent, IQLConfig
from hqa_orl.minari_io import EpisodeRecord, load_npz_buffer, records_to_npz


class CoreContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def batch(self):
        return {
            'obs': torch.zeros(2, 1), 'next_obs': torch.ones(2, 1),
            'actions': torch.tensor([[0.0], [0.5]]),
            'rewards': torch.tensor([[1.0], [3.0]]),
            'dones': torch.tensor([[0.0], [1.0]]),
            'weights': torch.tensor([[1.0], [3.0]]),
            'support': torch.tensor([[1.0], [0.0]]),
            'uncertainty': torch.tensor([[0.0], [1.0]]),
        }

    def test_hqa_loss_uses_weighted_mean_and_unormalized_bc_coefficient(self):
        cfg = IQLConfig(hidden_dim=4, hidden_depth=1, actor_lr=0,
                        value_lr=0, critic_lr=0, lambda_base=.2,
                        lambda_rho_uncertainty=1, lambda_eta_support=.5)
        agent = IQLAgent(1, 1, cfg, torch.device('cpu'))
        # Zero networks: Q=V=0, standard Normal before tanh, advantage weight=1.
        with torch.no_grad():
            for net in (agent.actor, agent.q, agent.target_q, agent.v):
                for parameter in net.parameters():
                    parameter.zero_()
        batch = self.batch()
        actions = np.array([0.0, .5], dtype=np.float32)
        pre_tanh = np.arctanh(actions)
        nll = .5 * pre_tanh**2 + .5 * np.log(2*np.pi) + np.log(1-actions**2 + 1e-6)
        weighted_nll = (nll[0] + 3*nll[1]) / (4 + 1e-6)
        # lambda=[.1,.4]; added regularizer is a batch mean, not divided by sum(lambda).
        regularizer = (.1*nll[0] + .4*nll[1]) / 2
        result = agent.update(batch)
        self.assertAlmostEqual(result['loss/value'], 0.0, places=7)
        self.assertAlmostEqual(result['loss/q'], (2*1 + 3*2*9)/(4+1e-6), places=5)
        self.assertAlmostEqual(result['loss/actor'], weighted_nll + regularizer, places=5)
        self.assertAlmostEqual(result['stats/lambda_mean'], .25, places=6)

    def test_behavior_cloning_ignores_value_and_quality_fields(self):
        cfg = BCConfig(hidden_dim=4, hidden_depth=1)
        first = BCAgent(1, 1, cfg, torch.device('cpu'))
        second = BCAgent(1, 1, cfg, torch.device('cpu'))
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        batch = self.batch()
        other = copy.deepcopy(batch)
        for key in ('weights', 'support', 'uncertainty', 'rewards', 'dones', 'next_obs'):
            other[key].fill_(float('nan'))
        self.assertEqual(first.update(batch), second.update(other))
        for a, b in zip(first.actor.parameters(), second.actor.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_serialization_requires_equal_dimensions_and_pickle_free_metadata(self):
        def record(dim):
            return EpisodeRecord(obs=np.zeros((2, dim), np.float32),
                next_obs=np.ones((2, dim), np.float32), actions=np.zeros((2, 1), np.float32),
                rewards=np.ones(2, np.float32), dones=np.zeros(2, np.float32),
                truncations=np.array([0, 1], np.float32), returns_to_go=np.array([1.99, 1], np.float32),
                episode_return=2, episode_length=2, source_name='fixture', source_prior=.5,
                source_dataset_id='fixture-v0', original_episode_id=7)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'data.npz')
            with self.assertRaisesRegex(ValueError, 'semantic alignment'):
                records_to_npz([record(2), record(3)], path, {})
            records_to_npz([record(2)], path, {'sample_count': np.int64(2)})
            data = load_npz_buffer(path)
            metadata = json.loads(str(data['metadata']))
            self.assertEqual(metadata['obs_alignment'], 'equal_dimensions_required')
            self.assertEqual(metadata['sample_count'], 2)
            np.testing.assert_array_equal(data['obs'], np.zeros((2, 2), np.float32))


if __name__ == '__main__':
    unittest.main()

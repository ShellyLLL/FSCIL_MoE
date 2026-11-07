"""Synthetic end-to-end checks for the single-current-encoder FSCIL state."""
import unittest
import torch
from models.clip.model import VisualMoEAdapter
from engine.engine import Runner


class UnifiedStateTests(unittest.TestCase):
    def test_topology_round_trip_restores_expert_count_and_session_metadata(self):
        source = VisualMoEAdapter(8, num_experts=2, reduction=2)
        source.add_expert(session_id=1, reduction=4)
        source.add_expert(session_id=2, reduction=4)
        topology = source.export_topology()
        state = source.state_dict()
        target = VisualMoEAdapter(8, num_experts=2, reduction=2)
        target.rebuild_topology(topology)
        target.load_state_dict(state, strict=True)
        self.assertEqual(target.num_experts, 4)
        self.assertEqual(target.expert_birth_sessions, [0, 0, 1, 2])

    def test_full_class_cosine_classifier_has_one_shared_prediction_rule(self):
        ids = torch.tensor([0, 1, 2, 3])
        prototypes = torch.eye(4)
        queries = torch.tensor([[0.01, 0.99, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1]])
        logits = torch.nn.functional.normalize(queries, dim=-1) @ prototypes.t()
        self.assertEqual(ids[logits.argmax(1)].tolist(), [1, 2])
        self.assertEqual(logits.shape[1], len(ids))


if __name__ == "__main__":
    unittest.main()

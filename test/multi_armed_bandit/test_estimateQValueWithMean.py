from unittest import TestCase
from src.multi_armed_bandit.EstimateQValueWithMean import EstimateQValueWithMean


class TestEstimateQValueWithMean(TestCase):
    def test_estimate_q_value(self):
        action_map_qobj = {
            1: {'q_value': 5, 'n': 0}
        }
        qobj = action_map_qobj[1]
        EstimateQValueWithMean.estimate_q_value(reward=1, q_obj=qobj)

        self.assertEqual(qobj, {'q_value': 1, 'n': 1})
        self.assertDictEqual(action_map_qobj, {1: {'q_value': 1, 'n': 1}})


        EstimateQValueWithMean.estimate_q_value(reward=4, q_obj=qobj)

        self.assertEqual(qobj, {'q_value': 2.5, 'n': 2})
        self.assertDictEqual(action_map_qobj, {1: {'q_value': 2.5, 'n': 2}})




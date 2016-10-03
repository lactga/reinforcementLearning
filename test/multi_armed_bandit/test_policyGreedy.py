from unittest import TestCase

from src.explorers.Greedy import Greedy


class TestPolicyGreedy(TestCase):
    def test_select_action(self):
        action_map_qobj = {
            1: {'q_value': 5},
            2: {'q_value': 3}
        }
        action = Greedy.select_action(action_map_qobj=action_map_qobj)
        self.assertEqual(action, 1)

    def test_get_action_with_max_q_value(self):
        action_map_qobj = {
            1: {'q_value': 1},
            2: {'q_value': 3}
        }
        action = Greedy.get_action_with_max_q_value(action_map_qobj)
        self.assertEqual(action, 2)

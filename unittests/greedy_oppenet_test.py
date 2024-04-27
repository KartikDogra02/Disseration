import unittest
import chess
from opponent_agent import GreedyOpponent, evaluate_board

class TestGreedyOpponent(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.agent = GreedyOpponent()

    def test_init(self):
        # Test initialization (if there's anything specific to test)
        self.assertIsInstance(self.agent, GreedyOpponent)

    def test_choose_action(self):
        # This assumes at least one legal move is available
        move = self.agent.choose_action(self.board)
        self.assertIn(move, list(self.board.legal_moves))

    def test_evaluate_board(self):
        # Basic test to check if evaluation is returning a number
        value = evaluate_board(self.board)
        self.assertIsInstance(value, (int, float))

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import chess
from state_representation import CustomStateRepresentation

class TestCustomStateRepresentation(unittest.TestCase):
    def setUp(self):
        self.state_rep = CustomStateRepresentation()
        self.board = chess.Board()

    def test_initial_conditions(self):
        # Test initial configuration
        self.assertEqual(self.state_rep.board_size, 8)
        self.assertEqual(self.state_rep.num_channels, 16)

    def test_get_state(self):
        # Test the state representation output dimensions
        state = self.state_rep.get_state(self.board)
        expected_shape = (self.state_rep.board_size, self.state_rep.board_size, self.state_rep.num_channels)
        self.assertEqual(state.shape, expected_shape)
        # Test values at initial board setup for a specific piece
        # For example, a white pawn on e2 at the start of the game
        e2_index = 8 * 6 + 4  # 6th row, 4th column in the array
        self.assertTrue(state[6][4][0] > 0)  # Assuming the index for pawns is 0

if __name__ == '__main__':
    unittest.main()

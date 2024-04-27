import unittest
import chess
from piece_stratergy import PawnStrategy, KnightStrategy, BishopStrategy, RookStrategy, QueenStrategy, KingStrategy

class TestPieceStrategies(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()

    def test_pawn_strategy(self):
        strategy = PawnStrategy(self.board, chess.PAWN)
        move = chess.Move.from_uci('e2e4')
        score = strategy.evaluate_move(move)
        self.assertEqual(score, 10)  # Assuming controlling the center gives a score of 10

    def test_knight_strategy(self):
        strategy = KnightStrategy(self.board, chess.KNIGHT)
        move = chess.Move.from_uci('g1f3')
        score = strategy.evaluate_move(move)
        self.assertEqual(score, 20)  # Assuming control of central squares adds 20

    def test_bishop_strategy(self):
        strategy = BishopStrategy(self.board, chess.BISHOP)
        move = chess.Move.from_uci('c1f4')
        score = strategy.evaluate_move(move)
        self.assertTrue(score > 0)  # Test specific score logic as defined

    def test_rook_strategy(self):
        strategy = RookStrategy(self.board, chess.ROOK)
        move = chess.Move.from_uci('a1d1')
        score = strategy.evaluate_move(move)
        self.assertTrue(score > 0)  # Test specific score logic as defined

    def test_queen_strategy(self):
        strategy = QueenStrategy(self.board, chess.QUEEN)
        move = chess.Move.from_uci('d1d3')
        score = strategy.evaluate_move(move)
        self.assertTrue(score > 0)  # Test specific score logic as defined

    def test_king_strategy(self):
        strategy = KingStrategy(self.board, chess.KING)
        move = chess.Move.from_uci('e1e2')
        score = strategy.evaluate_move(move)
        self.assertTrue(score > 0)  # Test specific score logic as defined

if __name__ == '__main__':
    unittest.main()


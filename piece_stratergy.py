import chess

class PieceStrategy:
    def __init__(self, board, piece_type):
        self.board = board
        self.piece_type = piece_type

    def evaluate_move(self, move):
        """ Evaluate the strategic value of a move. """
        raise NotImplementedError("This method should be overridden by subclasses.")

class PawnStrategy(PieceStrategy):
    def evaluate_move(self, move):
        # Pawns are encouraged to advance, control center, and promote
        score = 0
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 10  # Control the center
        if move.promotion:
            score += 100  # Promotion potential
        return score

class KnightStrategy(PieceStrategy):
    def evaluate_move(self, move):
        # Knights are valued for their ability to fork and control central squares
        score = 0
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 20  # Central control
        # Further implementation can include checking for forks
        return score

class BishopStrategy(PieceStrategy):
    def evaluate_move(self, move):
        # Bishops value long diagonals and staying on opposite color of opponent's king
        score = 0
        # Diagonal dominance (example simplification)
        score += len(self.board.attacks(move.to_square))
        return score

class RookStrategy(PieceStrategy):
    def evaluate_move(self, move):
        score = 0
        if is_open_file(self.board, move.to_square):
            score += 30  # Bonus for moving to an open file
        return score

class QueenStrategy(PieceStrategy):
    def evaluate_move(self, move):
        # Queens value freedom of movement and creating threats
        score = 0
        score += len(self.board.attacks(move.to_square))  # The more attacks from the square, the better
        return score

class KingStrategy(PieceStrategy):
    def evaluate_move(self, move):
        # Kings prioritize safety in the opening and mobility in the endgame
        score = 0
        if self.board.is_endgame():
            score += chess.square_distance(move.from_square, move.to_square)
        # Safety checks can include not moving into check or avoiding open lines
        return score

def choose_strategy(board, piece_type):
    strategies = {
        chess.PAWN: PawnStrategy(board, piece_type),
        chess.KNIGHT: KnightStrategy(board, piece_type),
        chess.BISHOP: BishopStrategy(board, piece_type),
        chess.ROOK: RookStrategy(board, piece_type),
        chess.QUEEN: QueenStrategy(board, piece_type),
        chess.KING: KingStrategy(board, piece_type)
    }
    return strategies[piece_type]

def is_open_file(board, square):
    file_index = chess.square_file(square)
    for rank in range(8):  # Loop through all ranks
        if board.piece_at(chess.square(file_index, rank)) is not None:
            return False  # Found a piece in the file
    return True

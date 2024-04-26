import numpy as np
import chess

class CustomStateRepresentation:
    def __init__(self):
        self.board_size = 8
        self.num_channels = 16  # Increased number of channels
        self.piece_values = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                             'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}

    def get_state(self, board):
        state = np.zeros((self.board_size, self.board_size, self.num_channels))
        piece_map = board.piece_map()

        # 1. Encoding piece positions with sign indicating piece color
        for square, piece in piece_map.items():
            piece_value = self.piece_values[piece.symbol()]
            row, col = square // 8, square % 8
            state[row, col, abs(piece_value) - 1] = np.sign(piece_value)

        # 2. Adding features for pawn structure (e.g., isolated pawns, pawn chains)
        for file_index in range(len(chess.FILE_NAMES)):
            file_name = chess.FILE_NAMES[file_index]
            for rank in range(1, 8):
                square = chess.square(file_index, rank)
                if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
                    # Check if pawn is isolated
                    if not board.is_pinned(board.turn, square) and not board.is_attacked_by(not board.turn, square):
                        state[file_index, rank, 12] = 1
                    # Check if pawn is part of a pawn chain
                    elif (board.piece_at(chess.square(file_index - 1, rank - 1)) and
                          board.piece_at(chess.square(file_index + 1, rank - 1))):
                        state[file_index, rank, 13] = 1

        # 3. Adding features for king safety (e.g., pawn shelter, open lines near the king)
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        if white_king_square:
            white_king_row, white_king_col = white_king_square // 8, white_king_square % 8
            # Example: Check if squares in front of white king are empty
            if white_king_row > 0 and not board.piece_at(chess.square(white_king_col, white_king_row - 1)):
                state[white_king_row - 1, white_king_col, 14] = 1
        if black_king_square:
            black_king_row, black_king_col = black_king_square // 8, black_king_square % 8
            # Example: Check if squares in front of black king are empty
            if black_king_row < 7 and not board.piece_at(chess.square(black_king_col, black_king_row + 1)):
                state[black_king_row + 1, black_king_col, 15] = -1

        return state

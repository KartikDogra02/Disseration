import chess

class GreedyOpponent:
    def __init__(self):
        pass

    def choose_action(self, board):
        legal_moves = list(board.legal_moves)
        best_move = None
        best_move_value = -float('inf')  # Initialize to negative infinity
        
        # Evaluate each legal move and choose the one with the highest immediate reward
        for move in legal_moves:
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(move)
            # Evaluate the board after the move
            move_value = evaluate_board(board_copy)
            # If the move has a higher value than the current best move, update the best move
            if move_value > best_move_value:
                best_move = move
                best_move_value = move_value
        
        return best_move

def evaluate_board(board):
    """
    Evaluate the board position using multiple factors.
    """
    # Material advantage
    material_advantage = evaluate_material(board)

    # Mobility advantage
    mobility_advantage = evaluate_mobility(board)

    # King safety
    king_safety = evaluate_king_safety(board)

    # Pawn structure
    pawn_structure = evaluate_pawn_structure(board)

    # Combine the evaluations with weights
    evaluation = material_advantage + mobility_advantage + king_safety + pawn_structure

    return evaluation

def evaluate_material(board):
    """
    Evaluate the material advantage of white over black.
    """
    material_advantage = sum(piece_value(piece) for piece in board.piece_map().values() if piece.color == chess.WHITE)
    material_advantage -= sum(piece_value(piece) for piece in board.piece_map().values() if piece.color == chess.BLACK)
    return material_advantage

def evaluate_mobility(board):
    """
    Evaluate the mobility advantage of white over black.
    """
    # Count the number of legal moves for white and black
    white_mobility = len(list(board.legal_moves))
    board.turn = not board.turn  # Switch turns
    black_mobility = len(list(board.legal_moves))
    board.turn = not board.turn  # Switch back to original turn
    return white_mobility - black_mobility

def evaluate_king_safety(board):
    """
    Evaluate the safety of the king position.
    """
    safety_score = 0

    # Get the squares around the black king
    black_king_square = board.king(chess.BLACK)
    if black_king_square:
        for square in board.attackers(chess.WHITE, black_king_square):
            piece = board.piece_at(square)
            if piece:
                safety_score -= piece_value(piece)

    return safety_score

def evaluate_pawn_structure(board):
    """
    Evaluate the pawn structure.
    """
    pawn_score = 0

    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE and piece.piece_type == chess.PAWN:
            # Bonus for pawns in the center
            if square in [chess.D4, chess.E4]:
                pawn_score += 0.5
            # Penalty for isolated pawns
            if not board.is_pinned(chess.WHITE, square) and not board.is_attacked_by(chess.BLACK, square):
                if not board.piece_at(square + 1) and not board.piece_at(square - 1):
                    pawn_score -= 0.5
            # Penalty for doubled pawns
            if board.piece_at(square - 8) and board.piece_at(square - 8).piece_type == chess.PAWN:
                pawn_score -= 0.5

    return pawn_score


def piece_value(piece):
    """
    Assign a value to each piece type for evaluation purposes.
    """
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type == chess.KNIGHT:
        return 3
    elif piece.piece_type == chess.BISHOP:
        return 3
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9
    elif piece.piece_type == chess.KING:
        return 100
    else:
        return 0  # Unknown piece type

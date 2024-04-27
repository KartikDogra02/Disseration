import chess
import random
import numpy as np
from rl_agent import ModelBasedRLAgent
from opponent_agent import GreedyOpponent
import threading
from piece_stratergy import choose_strategy  # Importing from piece_strategy.py
import matplotlib.pyplot as plt

class OpponentThread(threading.Thread):
    def __init__(self, opponent, board):
        super().__init__()
        self.opponent = opponent
        self.board = board
        self.move = None

    def run(self):
        self.move = self.opponent.choose_action(self.board)

def train_agents(episodes):
    agents = {}

    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        agents[piece_type] = ModelBasedRLAgent(piece_type)
    
    opponent = GreedyOpponent()  # Initialize the opponent agent

    white_wins = 0
    black_wins = 0
    draws = 0

    episode_rewards = []  # List to store rewards received in each episode

    for episode in range(episodes):
        board = chess.Board()
        total_reward = 0

        while not board.is_game_over():
            current_color = board.turn
            piece_type = choose_piece_type(board)
            agent = agents[piece_type]

            if current_color == chess.BLACK:
                opponent_thread = OpponentThread(opponent, board)
                opponent_thread.start()
                opponent_thread.join(timeout=5)
                if opponent_thread.move:
                    action = opponent_thread.move.uci()
                else:
                    action = random.choice([move.uci() for move in board.legal_moves])
            else:
                action = agent.choose_action(board)

            board.push_uci(action)

            reward = evaluate_reward(board, piece_type)
            agent.train(board, reward, agents.values())

            total_reward += reward

        episode_rewards.append(total_reward)

        game_result = get_game_result(board)
        if game_result == "white":
            white_wins += 1
        elif game_result == "black":
            black_wins += 1
        else:
            draws += 1
            
        print(f"Episode {episode + 1} complete Game result: {game_result}")

    print(f"White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
    print("Average reward per episode:", np.mean(episode_rewards))
    print("Max reward per episode:", np.max(episode_rewards))
    print("Min reward per episode:", np.min(episode_rewards))
    print("Standard deviation of rewards:", np.std(episode_rewards))

    # Plot the rewards received per episode
    plt.plot(range(1, episodes + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()
    return agents

def evaluate_reward(board, piece_type):
    """
    Evaluate the reward based on the current state of the board and the piece type.
    """
    # Initialize reward
    reward = 0  

    # Check the game result
    result = get_game_result(board)
    if result == "1-0":  # White wins
        reward += 1000  # High reward for White win
    elif result == "0-1":  # Black wins
        reward -= 1000  # Heavily penalize Black win
    elif result == "1/2-1/2":  # Draw
        reward -= 500  # Penalize draw

    # Strategic evaluation based on piece type
    if piece_type == chess.PAWN:
        reward += evaluate_pawn_advancement(board)
    elif piece_type == chess.KNIGHT:
        reward += evaluate_knight_moves(board)
    elif piece_type == chess.BISHOP:
        reward += evaluate_bishop_moves(board)
    elif piece_type == chess.ROOK:
        reward += evaluate_rook_moves(board)
    elif piece_type == chess.QUEEN:
        reward += evaluate_queen_moves(board)
    elif piece_type == chess.KING:
        reward += evaluate_king_moves(board)

    return reward


# Define helper functions for evaluating moves for each piece type
def evaluate_pawn_advancement(board):
    # Reward pawn advancement towards promotion or controlling key squares
    reward = 0

    result = get_game_result(board)
    if result == "1-0":  # White wins
        reward += 1000  # High reward for White win
    elif result == "0-1":  # Black wins
        reward -= 1000  # Heavily penalize Black win
    elif result == "1/2-1/2":  # Draw
        reward -= 600  # Penalize draw
    # Check if the pawn is in a position to promote
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                if chess.square_rank(square) == 6:
                    reward += 50  # Reward for a white pawn reaching the 7th rank
            else:
                if chess.square_rank(square) == 1:
                    reward += 50  # Reward for a black pawn reaching the 2nd rank
    # Control key squares in the center
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for square in center_squares:
        if board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
            reward += 5  # Reward for controlling central squares with white pawns
        elif board.piece_at(square) and board.piece_at(square).color == chess.BLACK:
            reward -= 5  # Penalize opponent's control of central squares
    return reward

def evaluate_knight_moves(board):
    # Reward knight moves that control central squares or target opponent's pieces
    reward = 0
    for move in board.legal_moves:
        if board.piece_at(move.to_square) and board.piece_at(move.to_square).color != board.turn:
            reward += 10  # Reward for targeting opponent's pieces
        if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
            reward += 5  # Reward for controlling central squares
    return reward

def evaluate_bishop_moves(board):
    # Reward bishop moves that control diagonals or target opponent's pieces
    reward = 0
    for move in board.legal_moves:
        if board.piece_at(move.to_square) and board.piece_at(move.to_square).color != board.turn:
            reward += 10  # Reward for targeting opponent's pieces
        # Reward for controlling long diagonals
        if move.to_square in [chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8]:
            reward += 5
        if move.to_square in [chess.A8, chess.B7, chess.C6, chess.D5, chess.E4, chess.F3, chess.G2, chess.H1]:
            reward += 5
    return reward

def evaluate_rook_moves(board):
    # Reward rook moves that control files or ranks, or target opponent's pieces
    reward = 0
    for move in board.legal_moves:
        if board.piece_at(move.to_square) and board.piece_at(move.to_square).color != board.turn:
            reward += 10  # Reward for targeting opponent's pieces
        # Reward for moving to open files or ranks
        if is_open_file(board, move.to_square):
            reward += 5
    return reward

def evaluate_queen_moves(board):
    # Reward queen moves that control files, ranks, or diagonals, or target opponent's pieces
    reward = 0
    for move in board.legal_moves:
        if board.piece_at(move.to_square) and board.piece_at(move.to_square).color != board.turn:
            reward += 10  # Reward for targeting opponent's pieces
        # Reward for moving to open files, ranks, or diagonals
        if is_open_file(board, move.to_square) or is_open_rank(board, move.to_square) or is_open_diagonal(board, move.to_square):
            reward += 5
    return reward

def evaluate_king_moves(board):
    # Reward king moves that improve king safety or target opponent's pieces
    reward = 0
    for move in board.legal_moves:
        if board.piece_at(move.to_square) and board.piece_at(move.to_square).color != board.turn:
            reward += 10  # Reward for targeting opponent's pieces
    return reward

# Helper function to check if a file is open (no pieces in the file)
def is_open_file(board, square):
    file_index = chess.square_file(square)
    for rank in range(8):
        if board.piece_at(chess.square(file_index, rank)):
            return False
    return True

# Helper function to check if a rank is open (no pieces in the rank)
def is_open_rank(board, square):
    rank_index = chess.square_rank(square)
    for file in range(8):
        if board.piece_at(chess.square(file, rank_index)):
            return False
    return True

# Helper function to check if a diagonal is open (no pieces in the diagonal)
def is_open_diagonal(board, square):
    for diag_square in chess.Diagonal(square):
        if board.piece_at(diag_square):
            return False
    return True


# The rest of your code (choose_piece_type, get_game_result, etc.) remains unchanged.

def choose_piece_type(board):
    """
    Choose the piece type to move based on a combination of evaluation and strategy.
    """
    # Define piece values for evaluation
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

    # Strategy-based piece selection
    if len(board.move_stack) < 10:
        # In the opening phase, prioritize control with pawns and centralization of knights
        if board.turn == chess.WHITE:
            return chess.PAWN
        else:
            return random.choice([chess.KNIGHT, chess.BISHOP])
    else:
        # Initialize piece mobility and safety dictionaries
        piece_mobility = {piece_type: 0 for piece_type in piece_values.keys()}
        piece_safety = {piece_type: 0 for piece_type in piece_values.keys()}

        # Calculate piece mobility and safety
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == chess.WHITE:
                piece_mobility[piece.piece_type] += 1
                if board.is_attacked_by(chess.BLACK, move.to_square):
                    piece_safety[piece.piece_type] -= 1
                else:
                    piece_safety[piece.piece_type] += 1

        # Calculate the weighted score for each piece type
        weighted_scores = {piece_type: piece_values[piece_type] + piece_mobility[piece_type] + piece_safety[piece_type] for piece_type in piece_values.keys()}

        # Decide based on the piece type with the highest weighted score
        best_piece = max(weighted_scores, key=weighted_scores.get)

        return best_piece

def get_game_result(board):
    if board.is_checkmate():
        winner_color = "white" if board.turn == chess.BLACK else "black"
        return winner_color
    elif board.is_stalemate():
        return "draw"
    else:
        return None

def play_against_agents(agents):
    board = chess.Board()
    print(board)
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print(board.legal_moves)
            while True:
                try:
                    human_move = input("Enter your move (in UCI format): ")
                    board.push_uci(human_move)
                    break  # Break the loop if input is valid
                except ValueError:
                    print("Invalid move format. Please try again.")
                except chess.InvalidMove:
                    print("Invalid move. Please try again.")
        else:
            piece_type = choose_piece_type(board)
            agent = agents[piece_type]
            action = agent.choose_action(board)
            board.push_uci(action)
        print(board)
    
    result = get_game_result(board)
    if result == "white":
        print("You win!")
    elif result == "black":
        print("You lose!")
    else:
        print("It's a draw!")



if __name__ == "__main__":
    trained_agents=train_agents(episodes=400)
    play_against_agents(trained_agents)

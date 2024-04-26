import chess
import random
from rl_agent import ModelBasedRLAgent
from opponent_agent import GreedyOpponent
import threading
import numpy as np
from pygame_chess_api.api import Board, Piece
from pygame_chess_api.render import Gui
import matplotlib.pyplot as plt
from stock_fish_api import StockFishOnlineOpponent


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
    other_agents = list(agents.values())  # Define other_agents list

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

            # If it's the opponent's turn, choose action using the opponent agent
            if current_color == chess.BLACK:
                opponent_thread = OpponentThread(opponent, board)
                opponent_thread.start()
                opponent_thread.join(timeout=5)  # Wait for opponent move with a timeout
                if opponent_thread.move:
                    action = opponent_thread.move.uci()  # Convert Move object to UCI string
                else:
                    action = random.choice([move.uci() for move in board.legal_moves])
            else:
                action = agent.choose_action(board)

            board.push_uci(action)

            # Evaluate the reward and train the agent
            reward = evaluate_reward(board)
            agent.train(board, reward, other_agents)

            total_reward += reward

        episode_rewards.append(total_reward)  # Store total reward for the episode

        game_result = get_game_result(board)
        if game_result == "white":
            white_wins += 1
        elif game_result == "black":
            black_wins += 1
        else:
            draws += 1
            
        print(f"Episode {episode + 1} complete Game result: {game_result}")

        # Communication between agents after each episode
        for agent in agents.values():
            agent.communicate_with_other_agents(other_agents)

    print(f"White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
    return agents

    # Calculate and print analytics
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

def evaluate_reward(board):
# Check if the game is over
    if board.is_game_over():
        result = board.result()
        if result == '1-0':
            # White wins
            return 1.0
        elif result == '0-1':
            # Black wins
            return -1.0
        else:
            # Draw
            return 0.5

    # If the game is not over, return a small negative reward to encourage faster wins
    return -0.01

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
    trained_agents=train_agents(episodes=10)
    play_against_agents(trained_agents)

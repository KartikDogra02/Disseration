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
    # Initialize an agent for each type of white piece
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
            if current_color == chess.WHITE:
                # Choose the piece type and corresponding agent based on the board state
                piece_type = choose_piece_type(board)
                agent = agents[piece_type]
                action = agent.choose_action(board)
                board.push_uci(action)
                reward = evaluate_reward(board)
                agent.train(board, reward, other_agents)
            else:
                action = opponent.choose_action(board)
                board.push_uci(action)
                # Optionally evaluate and update for learning purposes, if black also learns
                reward = evaluate_reward(board)
                opponent.train(board, reward, other_agents)  # Assuming opponent can also train

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
            
        print("Episode {} complete Game result: {}".format(episode + 1, game_result))


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
    if board.is_endgame():
        # In the endgame, prioritize king and queens for mobility and checkmate threats
        return chess.KING if board.piece_at(board.king(chess.WHITE)) else chess.QUEEN

    # Define piece values for evaluation
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0  # King's value can be context-specific
    }

    # Calculate mobility and position value for each piece
    piece_scores = {piece_type: 0 for piece_type in piece_values.keys()}
    for move in board.legal_moves:
        moved_piece = board.piece_at(move.from_square).piece_type
        piece_scores[moved_piece] += 1  # Increase score for mobility

        # Add positional bonuses
        if moved_piece == chess.KNIGHT and move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            piece_scores[moved_piece] += 2  # Central position bonus for knights

    # Adjust scores based on tactical situations
    # Example: Check for pins or forks that can be exploited or need to be prevented
    for move in board.legal_moves:
        if board.gives_check(move):
            piece_scores[board.piece_at(move.from_square).piece_type] += 5  # Bonus for checking moves

    # Decide based on the piece type with the highest score
    best_piece = max(piece_scores, key=piece_scores.get)

    return best_piece

def is_endgame(board):
    """
    Simple heuristic to determine if the game is in the endgame phase.
    """
    total_pieces = len(board.piece_map())
    return total_pieces <= 12  # Consider it endgame if there are 12 or fewer pieces on the board





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

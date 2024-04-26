import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten
import chess
from state_representation import CustomStateRepresentation
from collections import deque
import random

class ModelBasedRLAgent:
    def __init__(self, piece_type, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, communication_frequency=10, memory_capacity=1000):
        self.piece_type = piece_type
        self.state_representation = CustomStateRepresentation()
        self.input_shape = (8, 8, 16)  # Assuming your state representation is a 8x8x16 tensor
        self.output_shape = 64  # Output shape for 8x8 chessboard
        self.model = self.build_model()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.communication_frequency = communication_frequency
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=self.memory_capacity)  # Experience replay memory
        self.episode_counter = 0
        self.last_state = None
        self.last_action = None

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))  # Flatten the input
        model.add(Dense(512, activation='relu'))  # Increase units in the dense layer
        model.add(Dense(256, activation='relu'))  # Additional dense layer
        model.add(Dense(128, activation='relu'))  # Additional dense layer
        model.add(Dense(self.output_shape, activation='linear'))  # Output layer with linear activation

        model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

        return model

    def get_state_key(self, board):
        state = self.state_representation.get_state(board)
        return state

    def choose_action(self, board):
        state = self.get_state_key(board)
        if np.random.rand() < self.exploration_rate:
            # Explore: choose a random legal move
            legal_moves = [move.uci() for move in board.legal_moves]
            action = np.random.choice(legal_moves)
        else:
            # Exploit: choose action with highest Q-value from the neural network
            q_values = self.model.predict(np.array([state]))[0]
            legal_moves = [move.uci() for move in board.legal_moves]
            q_values_legal = {chess.Move.from_uci(move): q_values[chess.Move.from_uci(move).from_square] for move in legal_moves}
            action = max(q_values_legal, key=q_values_legal.get).uci()
        
        self.last_state = state
        self.last_action = action
        return action

    def update_memory(self, experiences):
        for experience in experiences:
            self.memory.append(experience)

    def sample_from_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def update_q_value(self, reward, next_state):
        if self.last_state is not None and self.last_action is not None:
            q_values_next = self.model.predict(np.array([next_state]))[0]
            max_q_value_next = np.max(q_values_next)
            target_q_values = np.zeros((1, self.output_shape))
            target_q_values[0] = self.model.predict(np.array([self.last_state]))[0]  # Copy current Q-values
            last_move = chess.Move.from_uci(self.last_action)
            last_square_index = last_move.from_square
            target_q_values[0, last_square_index] = reward + self.discount_factor * max_q_value_next  # Update Q-value
            self.model.fit(np.array([self.last_state]), target_q_values, epochs=1, verbose=0)

    def train(self, board, reward, other_agents):
        next_state = self.get_state_key(board)
        self.update_q_value(reward, next_state)

        # Store experience in memory
        self.memory.append((self.last_state, self.last_action, reward, next_state))

        self.episode_counter += 1
        if self.episode_counter % self.communication_frequency == 0:
            self.communicate_with_other_agents(other_agents)  # Periodically communicate with other agents

    def communicate_with_other_agents(self, other_agents):
        # Share experiences with other agents
        # Other agents can sample experiences from self.memory and update their own models
        # In this example, we just copy the memory to other agents
        for agent in other_agents:
            agent.update_memory(list(self.memory))

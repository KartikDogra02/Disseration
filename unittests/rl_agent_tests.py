import unittest
import numpy as np
import chess
from keras.src.models import Sequential
from rl_agent import ModelBasedRLAgent

class TestModelBasedRLAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ModelBasedRLAgent(chess.PAWN)  # Assuming it's designed for a specific piece type

    def test_initial_conditions(self):
        # Test if initial conditions like exploration_rate and memory_capacity are set correctly
        self.assertEqual(self.agent.exploration_rate, 0.1)
        self.assertEqual(self.agent.memory_capacity, 1000)

    def test_build_model(self):
        # Test if the model is built correctly
        self.assertIsInstance(self.agent.model, Sequential)
        self.assertEqual(len(self.agent.model.layers), 3)  # Assuming 3 layers in the model

    def test_model_output(self):
        # Test the model's prediction output shape
        input_shape = (1, *self.agent.input_shape)
        test_input = np.random.random(input_shape)
        output = self.agent.model.predict(test_input)
        self.assertEqual(output.shape, (1, self.agent.output_shape))

if __name__ == '__main__':
    unittest.main()

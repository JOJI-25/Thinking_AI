import torch
import random
from config import Config

class TextEnvironment:
    """
    Simulates a text stream for the agent to learn.
    """
    def __init__(self):
        # Simple Vocabulary: A, B, C...
        self.vocab_size = Config.INPUT_DIM
        self.seq_len = 100
        
        # synthetic grammar: "A B C" repeats
        self.pattern = [0, 1, 2, 3, 4] # "The cat sat on mat"
        
    def reset(self):
        self.idx = 0
        return self._get_obs(self.pattern[0])
        
    def _get_obs(self, token_id):
        # One-hot encoding
        obs = torch.zeros(1, self.vocab_size).to(Config.DEVICE)
        obs[0, token_id] = 1.0
        return obs
        
    def step(self, action_pred):
        """
        Action is the predicted token ID for the *Next* step.
        Reward is 1 if correct, -1 if wrong.
        """
        # Current true token was pattern[self.idx]
        # Next true token is pattern[self.idx + 1]
        
        next_idx = (self.idx + 1) % len(self.pattern)
        true_next_token = self.pattern[next_idx]
        
        if action_pred == true_next_token:
            reward = 1.0
        else:
            reward = -0.1
            
        self.idx = next_idx
        next_obs = self._get_obs(true_next_token)
        
        done = False # Continuous stream
        
        return next_obs, reward, done, true_next_token

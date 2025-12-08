import numpy as np
import torch
from config import Config

class GridWorld:
    """
    Simple 5x5 Grid World
    Goal: Bottom-right corner (4,4)
    Start: Top-left corner (0,0)
    Reward: +10 at goal, -0.1 step cost
    """
    def __init__(self, size=5):
        self.size = size
        self.state = np.array([0, 0], dtype=np.float32)
        self.goal = np.array([size-1, size-1], dtype=np.float32)
        self.max_steps = 50
        self.steps = 0
        
    def reset(self):
        self.state = np.array([0, 0], dtype=np.float32)
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        # Normalize coordinates to [-1, 1] usually helps neural nets
        # But here logic is simple 0..size
        # Let's return norm coords [0, 1]
        return torch.tensor(self.state / self.size, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        
    def step(self, action):
        """
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        """
        self.steps += 1
        
        y, x = self.state
        
        if action == 0:   # Up
            y = max(0, y - 1)
        elif action == 1: # Down
            y = min(self.size - 1, y + 1)
        elif action == 2: # Left
            x = max(0, x - 1)
        elif action == 3: # Right
            x = min(self.size - 1, x + 1)
            
        self.state = np.array([y, x], dtype=np.float32)
        
        # Check done
        dist = np.abs(self.state - self.goal).sum() 
        done = (dist == 0)
        
        # Reward
        if done:
            reward = 10.0
        else:
            reward = -0.1
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

class MemoryGridWorld:
    """
    Phase 2: Delayed Match-to-Sample Task
    Start: (0,0). Agent sees a 'Cue' (1 or 2).
    Goal 1: (0,4) - Correct if Cue was 1
    Goal 2: (4,0) - Correct if Cue was 2
    
    Observation: [y, x, cue_signal]
    - cue_signal is only visible at step 0.
    """
    def __init__(self, size=5):
        self.size = size
        self.state = np.array([0, 0], dtype=np.float32)
        self.cue = 0
        self.steps = 0
        self.max_steps = 30
        
    def reset(self):
        self.steps = 0
        self.state = np.array([0, 0], dtype=np.float32)
        # Random cue: 1 or 2
        self.cue = np.random.randint(1, 3) 
        
        # Obs: [norm_y, norm_x, cue, 0]
        # We need fixed input dim 4 to match config
        # Cue is visible ONLY at step 0
        return self._get_obs()
        
    def _get_obs(self):
        y, x = self.state
        norm_y = y / self.size
        norm_x = x / self.size
        
        # Visual cue is transient
        curr_cue = self.cue if self.steps == 0 else 0.0
        
        obs_vec = np.array([norm_y, norm_x, curr_cue, 0.0], dtype=np.float32)
        return torch.tensor(obs_vec, device=Config.DEVICE).unsqueeze(0)
        
    def step(self, action):
        self.steps += 1
        y, x = self.state
        
        if action == 0:   # Up
            y = max(0, y - 1)
        elif action == 1: # Down
            y = min(self.size - 1, y + 1)
        elif action == 2: # Left
            x = max(0, x - 1)
        elif action == 3: # Right
            x = min(self.size - 1, x + 1)
            
        self.state = np.array([y, x], dtype=np.float32)
        
        # Goals
        # Goal 1: Top-Right (0, 4) -> Matches Cue 1
        # Goal 2: Bottom-Left (4, 0) -> Matches Cue 2
        
        done = False
        reward = -0.05 # Step cost
        
        # Check Goal 1
        if y == 0 and x == self.size - 1:
            if self.cue == 1:
                reward = 10.0
            else:
                reward = -5.0 # Wrong goal
            done = True
            
        # Check Goal 2
        if y == self.size - 1 and x == 0:
            if self.cue == 2:
                reward = 10.0
            else:
                reward = -5.0 # Wrong goal
            done = True
            
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done

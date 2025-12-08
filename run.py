import torch
import numpy as np
import time
from config import Config
from environment import MemoryGridWorld
from agent import BioAgent
# Import classes for pickle loading is crucial when weights_only=False
from modules import Encoder, Predictor, ValueNetwork, PolicyNetwork, TransformerPredictor
from workspace import GlobalWorkspace
from neuron import LIFLayer
from plasticity import SynapseManager

def main():
    print("=== Thinking AI Launcher ===")
    print("Initializing Environment and Agent...")
    
    # 1. Setup Environment
    env = MemoryGridWorld(size=5)
    
    # 2. Load Agent Safely
    # PyTorch 2.6+ defaults to weights_only=True which breaks custom classes.
    # We explicitly allow full unpickling because we trust our own 'bio_agent_phase2.pth'.
    try:
        print("Loading 'bio_agent_phase2.pth'...")
        # Check if file exists first? torch.load will error if not found.
        try:
             agent = torch.load("bio_agent_phase2.pth", map_location=Config.DEVICE, weights_only=False)
             print(">> Agent loaded successfully!")
        except FileNotFoundError:
             print(">> Model 'bio_agent_phase2.pth' not found. Please run 'think.py' to train first.")
             return
             
    except Exception as e:
        print(f"CRITICAL ERROR loading agent: {e}")
        print("Tip: Ensure classes are imported and PyTorch version is compatible.")
        return

    # 3. Interactive / Demo Loop
    print("\nStarting Evaluation Run (5 Episodes)...")
    
    for ep in range(5):
        obs = env.reset()
        cue = env.cue
        agent.lif.reset_state(batch_size=1)
        agent.synapse_manager.reset_traces()
        agent.reset_history()
        
        print(f"\nEpisode {ep+1} Start | Cue: {cue}")
        
        done = False
        step = 0
        total_reward = 0
        
        # Initial 'Thought' (Action 0 default or deliberate)
        # We can use deliberate_action for the full 'Thinking' effect
        current_broadcast = None
        action, x_t, s_t, v_t, val_t, new_broadcast, ignition = agent.deliberate_action(obs, current_broadcast)
        
        while not done:
            # Step
            next_obs, reward, done = env.step(action)
            total_reward += reward
            step += 1
            
            if not done:
                # Next Thought
                next_broadcast_in = new_broadcast.detach()
                action, x_next, s_next, v_next, val_next, new_broadcast, ignition = agent.deliberate_action(next_obs, next_broadcast_in)
                obs = next_obs
                
            # Optional: Slow down for visual effect
            # time.sleep(0.1)

        status = "SUCCESS" if total_reward > 0 else "FAIL"
        print(f"Episode {ep+1} Finished: {status} (Reward: {total_reward:.2f})")

    print("\nExecution Complete.")

if __name__ == "__main__":
    main()

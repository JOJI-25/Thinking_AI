import torch
import numpy as np
import time
from config import Config
from environment import MemoryGridWorld
from agent import BioAgent
# Import classes for pickle loading
from modules import Encoder, Predictor, ValueNetwork, PolicyNetwork, TransformerPredictor
from workspace import GlobalWorkspace
from neuron import LIFLayer
from plasticity import SynapseManager

def test():
    print("Loading agent...")
    try:
        agent = torch.load("bio_agent_phase2.pth", map_location=Config.DEVICE, weights_only=False)
        # agent.eval() # BioAgent is not an nn.Module itself
    except Exception as e:
        print(f"Failed to load agent: {e}")
        return

    env = MemoryGridWorld(size=5)
    
    success_count = 0
    num_episodes = 20
    
    for ep in range(num_episodes):
        print(f"\n=== Test Episode {ep+1} ===")
        
        obs = env.reset()
        cue = env.cue
        
        agent.lif.reset_state(batch_size=1)
        agent.synapse_manager.reset_traces()
        agent.reset_history()
        
        # Phase 4: Use Deliberate Action (System 2)
        # First step:
        current_broadcast = None
        action, x_t, s_t, v_t, val_t, new_broadcast, ignition = agent.deliberate_action(obs, current_broadcast)
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Visualization (Sparse output)
            if ep < 3: 
                ignited = "IGNITION!" if ignition.item() > 0.5 else "."
                act_name = ["Up", "Down", "Left", "Right"][action]
                print(f"Step {step}: Act: {act_name} | {ignited}")
            
            # Step
            next_obs, reward, done = env.step(action)
            total_reward += reward
            step += 1
            
            if not done:
                next_broadcast_in = new_broadcast.detach()
                next_action, x_next, s_next, v_next, val_next, next_broadcast_out, next_ign = agent.deliberate_action(next_obs, next_broadcast_in)
                
                # Advance
                obs = next_obs
                action = next_action
                new_broadcast = next_broadcast_out
                ignition = next_ign
                
        if total_reward > 0:
            print(f">> Episode {ep+1} SUCCESS (Reward: {total_reward:.2f})")
            success_count += 1
        else:
            print(f">> Episode {ep+1} FAILED (Reward: {total_reward:.2f})")

    print(f"\nTest Complete. Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")

if __name__ == "__main__":
    test()

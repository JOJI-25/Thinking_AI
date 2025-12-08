import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment import GridWorld
from agent import BioAgent

def train():
    env = GridWorld(size=5)
    agent = BioAgent()
    
    num_episodes = 200
    rewards_history = []
    
    
    print("Starting training...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        agent.lif.reset_state(batch_size=1)
        agent.synapse_manager.reset_traces()
        
        # Run one step to prime s_0?
        # Actually GridWorld starts at t=0.
        # We process obs -> action -> env -> next_obs.
        
        # To make it work seamlessly:
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment import MemoryGridWorld
from agent import BioAgent

def train():
    env = MemoryGridWorld(size=5) # Phase 2 Env
    agent = BioAgent()
    
    num_episodes = 500 # More eps for memory task
    rewards_history = []
    
    
    print("Starting training...")
    
    # --- Correct Loop Logic with Variables ---
        
    for episode in range(num_episodes):
        obs = env.reset()
        agent.lif.reset_state(batch_size=1)
        agent.synapse_manager.reset_traces()
        agent.reset_history() # Phase 3
        
        current_broadcast = None
        
        # t=0
        action, x_t, s_t, v_t, val_t, new_broadcast, ignition = agent.get_action(obs, current_broadcast)
        
        total_reward = 0
        done = False
        
        while not done:
            # Step Env
            next_obs, reward, done = env.step(action)
            total_reward += reward
            
            # Context for gradient calc at t
            # We used 'current_broadcast' as input
            broadcast_in_t = current_broadcast
            
            # Prepare t+1
            if not done:
                next_broadcast_in = new_broadcast.detach() # Feedback from t acts on t+1
                next_action, x_next, s_next, v_next, val_next, next_broadcast_out, next_ign = agent.get_action(next_obs, next_broadcast_in)
                
                target_val = val_next.item()
                target_x = x_next.detach()
            else:
                target_val = 0.0
                with torch.no_grad():
                     target_x = agent.encoder(next_obs)
            
            # Learning
            td_target = reward + Config.GAMMA * target_val
            td_error = td_target - val_t.item()
            agent.synapse_manager.apply_weight_update(td_error)
            
            # Pass ACTION taken to forward_heads
            x_graph, val_graph, dist_graph, pred_graph = agent.forward_heads(obs, s_t, action, broadcast_in_t)
            
            loss_v = (val_graph - td_target).pow(2)
            loss_pred = (pred_graph - target_x).pow(2).mean()
            
            adv = td_error
            log_prob = dist_graph.log_prob(torch.tensor(action, device=Config.DEVICE))
            loss_pi = -log_prob * adv - 0.01 * dist_graph.entropy()
            
            # Auxiliary GWT Loss? 
            # Prompt doesn't require explicit GWT supervision, usually emergent or via RL.
            # But we want to encourage usage?
            loss = Config.LAMBDA_V * loss_v + Config.LAMBDA_PI * loss_pi + Config.LAMBDA_PRED * loss_pred
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            # Advance
            if not done:
                obs = next_obs
                action = next_action
                x_t, s_t, v_t, val_t = x_next, s_next, v_next, val_next
                new_broadcast = next_broadcast_out # For next step output
                current_broadcast = next_broadcast_in # The input used for next step
                
        rewards_history.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")

    print("Training Complete.")
    torch.save(agent, "bio_agent_phase2.pth")

if __name__ == "__main__":
    train()

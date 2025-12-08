import torch
import numpy as np
from config import Config
from text_environment import TextEnvironment
from agent import BioAgent

def train_generative():
    env = TextEnvironment()
    agent = BioAgent()
    
    # Force agent heads to match vocab size if needed
    # (Config.ACTION_DIM updated to 50 already)
    
    num_steps = 2000
    print(f"Training Generative Model on Vocab Size {Config.VOCAB_SIZE if hasattr(Config, 'VOCAB_SIZE') else Config.ACTION_DIM}...")
    
    obs = env.reset()
    agent.lif.reset_state(batch_size=1)
    agent.synapse_manager.reset_traces()
    agent.reset_history()
    
    current_broadcast = None
    
    # t=0
    # For generative task, "action" is the prediction of the next token.
    action, x_t, s_t, v_t, val_t, new_broadcast, ignition = agent.get_action(obs, current_broadcast)
    
    correct_count = 0
    
    for step in range(num_steps):
        # Env Step: Check if action (prediction) matched next token
        next_obs, reward, done, true_token = env.step(action)
        
        if reward > 0:
            correct_count += 1
            
        # Context
        broadcast_in_t = current_broadcast
        
        # Advance Agent
        next_broadcast_in = new_broadcast.detach()
        next_action, x_next, s_next, v_next, val_next, next_broadcast_out, next_ign = agent.get_action(next_obs, next_broadcast_in)
        
        # Target for Critic
        target_val = val_next.item()
        
        # Learning
        td_target = reward + Config.GAMMA * target_val
        td_error = td_target - val_t.item()
        agent.synapse_manager.apply_weight_update(td_error)
        
        # Forward Heads for Gradients
        # IMPORTANT: "Action" here is the predicted token ID
        x_graph, val_graph, dist_graph, pred_graph = agent.forward_heads(obs, s_t, action, broadcast_in_t)
        
        loss_v = (val_graph - td_target).pow(2)
        
        # Policy Loss (RL) - Reinforcing correct predictions
        adv = td_error
        try:
             log_prob = dist_graph.log_prob(torch.tensor(action, device=Config.DEVICE))
        except:
             # Fallback if action out of bounds (shouldn't happen if config syncs)
             log_prob = torch.tensor(0.0, device=Config.DEVICE)
             
        loss_pi = -log_prob * adv - 0.01 * dist_graph.entropy()
        
        # Pred Loss (World Model) - Predicting the embedding of the next token
        # next_obs is the One-Hot of the true next token
        with torch.no_grad():
            target_x = agent.encoder(next_obs)
            
        loss_pred = (pred_graph - target_x).pow(2).mean()
        
        loss = Config.LAMBDA_V * loss_v + Config.LAMBDA_PI * loss_pi + Config.LAMBDA_PRED * loss_pred
        
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        
        # Debug
        if step % 100 == 0:
            print(f"Step {step}: Reward {reward:.2f} | Acc {correct_count/100:.2f} | Validating: Predicted {action} vs True {true_token}")
            correct_count = 0
            
        # Advance variables
        obs = next_obs
        action = next_action
        x_t, s_t, v_t, val_t = x_next, s_next, v_next, val_next
        new_broadcast = next_broadcast_out
        current_broadcast = next_broadcast_in
        ignition = next_ign

    print("Generative Training Complete.")
    
    # Test Generation
    print("\n--- Generating Text ---")
    current_token = 0 # Start with '0'
    obs = env._get_obs(current_token)
    
    gen_text = [str(current_token)]
    
    for _ in range(20):
        # We use the agent's action (prediction) as the next input!
        # This is Autoregressive Generation
        action, x_t, s_t, v_t, val_t, new_broadcast, ignition = agent.get_action(obs, current_broadcast)
        
        gen_text.append(str(action))
        
        # Feed back prediction as input
        obs = env._get_obs(action)
        # Assuming simplified step without state reset for generation
        
    print("Generated Sequence:", " ".join(gen_text))
    print("Expected Pattern: 0 1 2 3 4 0 1 2 ...")

if __name__ == "__main__":
    train_generative()

import torch
import torch.optim as optim
from config import Config
from neuron import LIFLayer
from plasticity import SynapseManager
from modules import Encoder, Predictor, ValueNetwork, PolicyNetwork, TransformerPredictor

from workspace import GlobalWorkspace

class BioAgent:
    def __init__(self):
        # 1. Perception
        self.encoder = Encoder(Config.INPUT_DIM, Config.EMBED_DIM).to(Config.DEVICE)
        
        # 2. Spiking Core
        self.lif = LIFLayer(Config.EMBED_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
        self.lif.reset_state(batch_size=1)
        
        # 3. Plasticity
        self.synapse_manager = SynapseManager(self.lif, Config.EMBED_DIM, Config.HIDDEN_DIM)
import torch
import torch.optim as optim
from config import Config
from neuron import LIFLayer
from plasticity import SynapseManager
from modules import Encoder, Predictor, ValueNetwork, PolicyNetwork, TransformerPredictor

from workspace import GlobalWorkspace

class BioAgent:
    def __init__(self):
        # 1. Perception
        self.encoder = Encoder(Config.INPUT_DIM, Config.EMBED_DIM).to(Config.DEVICE)
        
        # 2. Spiking Core
        self.lif = LIFLayer(Config.EMBED_DIM, Config.HIDDEN_DIM).to(Config.DEVICE)
        self.lif.reset_state(batch_size=1)
        
        # 3. Plasticity
        self.synapse_manager = SynapseManager(self.lif, Config.EMBED_DIM, Config.HIDDEN_DIM)
        
        # 4. Global Workspace (Phase 2)
        self.workspace = GlobalWorkspace(Config.EMBED_DIM).to(Config.DEVICE)
        
        # 5. Heads
        # Phase 3: Transformer World Model
        self.predictor = TransformerPredictor(Config.EMBED_DIM, Config.HIDDEN_DIM, Config.ACTION_DIM,
                                             heads=Config.TRANSFORMER_HEADS, 
                                             layers=Config.TRANSFORMER_LAYERS).to(Config.DEVICE)
                                             
        self.value_net = ValueNetwork(Config.HIDDEN_DIM).to(Config.DEVICE)
        self.policy_net = PolicyNetwork(Config.HIDDEN_DIM, Config.ACTION_DIM).to(Config.DEVICE)
        
        # Optimizers (Added workspace params)
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.workspace.parameters()},
            {'params': self.predictor.parameters()},
            {'params': self.value_net.parameters()},
            {'params': self.policy_net.parameters()}
        ], lr=Config.LR_ACTOR_CRITIC)
        
        # Context Buffer for Transformer (Phase 3)
        self.history = []
        self.max_history = 10
        
    def reset_history(self):
        self.history = []
        
    def _make_feature(self, x, s, a):
        # Create (1, Dim) feature vector
        # a is integer
        a_onehot = torch.zeros(1, Config.ACTION_DIM, device=Config.DEVICE)
        a_onehot[0, a] = 1.0
        return torch.cat([x, s, a_onehot], dim=-1)

    def get_action(self, obs, prev_broadcast=None):
        """
        Executes Forward Dynamics.
        """
        if prev_broadcast is None:
            prev_broadcast = torch.zeros(1, Config.EMBED_DIM, device=Config.DEVICE)
            
        with torch.no_grad():
            x_in = self.encoder(obs)
            x_t = x_in + prev_broadcast
            
            s_t, v_t = self.lif(x_t)
            
            self.synapse_manager.update_traces(x_t, s_t)
            
            broadcast, ignition = self.workspace(x_t, s_t)
            
            probs = self.policy_net(s_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            value = self.value_net(s_t)
            
            # Store history for next steps
            # We store the State + Action taken
            feat = self._make_feature(x_t, s_t, action.item())
            self.history.append(feat)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
        return action.item(), x_t, s_t, v_t, value, broadcast, ignition

    def deliberate_action(self, obs, prev_broadcast=None):
        """
        Phase 4: System 2 Reasoning
        Simulates consequences of each action before deciding.
        """
        if prev_broadcast is None:
            prev_broadcast = torch.zeros(1, Config.EMBED_DIM, device=Config.DEVICE)
            
        with torch.no_grad():
            # 1. Current State Perception (same as get_action)
            x_in = self.encoder(obs)
            x_t = x_in + prev_broadcast
            s_t, v_t = self.lif(x_t) # Note: this updates true state if not careful? 
            # In 'think.py', we call get_action to update state.
            # Here, we want to update state ONLY ONCE.
            # So deliberate_action functions as get_action but with a pause.
            
            # REAL State Update
            # In a real agent, we might pause updates while thinking.
            # Here, we assume this call REPLACES get_action step.
            
            self.synapse_manager.update_traces(x_t, s_t)
            broadcast, ignition = self.workspace(x_t, s_t)
            
            # 2. System 1 (Fast) Proposal
            probs = self.policy_net(s_t)
            
            # 3. System 2 (Slow) Simulation
            # Only if Ignition is high? Or always for demo.
            print(f"\n[System 2] Thinking... (Ingition: {ignition.item():.2f})")
            
            best_val = -999.0
            best_act = 0
            
            # Construct current history context
            if len(self.history) > 0:
                hist_tensor = torch.stack(self.history)
            else:
                hist_tensor = None
                
            for a in range(Config.ACTION_DIM):
                # Hypothetical input: Current History + (x_t, s_t, a)
                curr_feat = self._make_feature(x_t, s_t, a)
                
                if hist_tensor is not None:
                    # Seq len + 1
                    sim_input = torch.cat([hist_tensor, curr_feat.unsqueeze(0)], dim=0)
                else:
                    sim_input = curr_feat.unsqueeze(0)
                    
                # Predict Future x_{t+1}
                pred_x_next = self.predictor(sim_input)
                
                # Estimate Value of Future
                # We need s_{t+1}. We approximate by running LIF on pred_x_next
                # But we must not change self.lif state!
                # We use a throwaway clone or just functional call?
                # LIFLayer has simple dynamics.
                # s_next approx = 1 if v + input > thr
                # This is a rough heuristic: V(Predictor(x,s,a))
                
                # Quick hack: pass predicted embedding to ValueNet directly?
                # ValueNet expects s_t.
                # Let's map pred_x_next -> "supposed spikes"?
                # or just use the current s_t? No, that misses the point.
                
                # Evaluation Heuristic:
                # Use Soft Spiking for robustness against prediction noise
                v_temp = self.lif.v.clone()
                i_temp = self.lif.fc(pred_x_next) 
                
                # Dynamics
                v_new = Config.ALPHA * v_temp + (1 - Config.ALPHA) * i_temp
                
                # Soft Spike: sigmoid((v - th) * scale)
                # Steepness 10.0
                s_new_soft = torch.sigmoid((v_new - Config.V_TH) * 10.0)
                
                val_est = self.value_net(s_new_soft).item()
                
                act_name = ["Up", "Down", "Left", "Right"][a]
                print(f"  Thought: Action '{act_name}' -> Pred Value {val_est:.2f}")
                
                if val_est > best_val:
                    best_val = val_est
                    best_act = a
            
            # Cognitive Synergy: Combine System 1 (Instinct) and System 2 (Reasoning)
            # If System 2 is flat (best_val is low), rely on System 1.
            
            if best_val < 0.1: # Heuristic threshold for "meaningful thought"
                 print("  [Thought] Uncertainty high... Relying on Instinct (System 1).")
                 final_action = torch.argmax(probs).item()
            else:
                 final_action = best_act
                 
            act_str = ['Up','Down','Left','Right'][final_action]
            print(f"[System 2] Decided: {act_str}")
            
            value = self.value_net(s_t)
            
            # Commit to history
            feat = self._make_feature(x_t, s_t, final_action)
            self.history.append(feat)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        return final_action, x_t, s_t, v_t, value, broadcast, ignition

    def forward_heads(self, obs, s_t, action_taken, prev_broadcast=None):
        """
        Re-runs forward pass for Gradient Calculation.
        Need history context for Transformer.
        action_taken: The action actually taken at step t (int)
        """
        if prev_broadcast is None:
            prev_broadcast = torch.zeros(1, Config.EMBED_DIM, device=Config.DEVICE)
            
        x_in = self.encoder(obs)
        x_t = x_in + prev_broadcast
        
        value = self.value_net(s_t)
        probs = self.policy_net(s_t)
        dist = torch.distributions.Categorical(probs)
        
        # Transformer Prediction
        # Construct history batch with the ACTUAL action taken appended
        if len(self.history) > 0:
            hist_tensor = torch.stack(self.history) 
            # The last element in history ALREADY contains the action taken (added in get_action)
            # But that was detached.
            # We need to rebuild the graph for the CURRENT step.
            
            # history[-1] is the feature for step t.
            # predictive loss is on x_{t+1}.
            # Input to transformer is sequence up to t.
            
            # Reconstruct current feature with gradients
            a_onehot = torch.zeros(1, Config.ACTION_DIM, device=Config.DEVICE)
            a_onehot[0, action_taken] = 1.0
            
            curr_feat = torch.cat([x_t, s_t, a_onehot], dim=-1)
            
            if len(self.history) > 1:
                past = hist_tensor[:-1].detach()
                seq = torch.cat([past, curr_feat.unsqueeze(0)], dim=0)
            else:
                seq = curr_feat.unsqueeze(0)
                
            pred_next = self.predictor(seq)
        else:
            pred_next = torch.zeros(1, Config.EMBED_DIM, device=Config.DEVICE)
        
        return x_t, value, dist, pred_next

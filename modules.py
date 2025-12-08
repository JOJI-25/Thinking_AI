import torch
import torch.nn as nn
from config import Config

class Encoder(nn.Module):
    """
    Perception / Encoder Module
    x_t = E_theta(o_t)
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.Tanh() # Enforce some bounds on embedding
        )
        
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    """
    World Model / Predictor
    x^_t+1 = f_theta(x_t, s_t)
    Prediction depends on current embedding and internal state (spikes).
    """
    def __init__(self, embed_dim, state_dim):
        super().__init__()
        # Input: Concatenation of x_t and s_t
        self.net = nn.Sequential(
            nn.Linear(embed_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, x_t, s_t):
        input_feat = torch.cat([x_t, s_t], dim=-1)
        return self.net(input_feat)

class ValueNetwork(nn.Module):
    """
    Value Function
    V(s_t) = V_phi(s_t)
    """
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, s_t):
        return self.net(s_t)

class PolicyNetwork(nn.Module):
    """
    Policy Distribution
    pi(a_t | s_t)
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, s_t):
        return self.net(s_t)

class TransformerPredictor(nn.Module):
    """
    Phase 3: Transformer World Model
    Predicts next embedding based on history window AND action.
    """
    def __init__(self, embed_dim, state_dim, action_dim, heads=2, layers=1):
        super().__init__()
        # Input: Embed + State + Action (One-hot)
        self.input_dim = embed_dim + state_dim + action_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
        self.decoder = nn.Linear(self.input_dim, embed_dim)
        self.action_dim = action_dim
        
    def forward(self, history, action=None):
        """
        history: (Sequence, Batch, InputDim - ActionDim) - Old style?
        Actually, we need history of (x, s, a).
        For simplicity in this phase, let's assume 'history' already contains actions?
        Or we append the current action to the history context to predict next?
        
        Let's treat 'backward' training: we pass history of (x,s,a).
        """
        # We assume input is (Seq, Batch, Dim).
        # Transformer output
        out = self.transformer(history)
        # Take last step output to predict next
        last_out = out[-1] 
        pred = self.decoder(last_out)
        return pred

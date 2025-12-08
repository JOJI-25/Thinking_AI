import torch
import torch.nn as nn
from config import Config

class GlobalWorkspace(nn.Module):
    """
    Phase 2: Global Workspace
    Eq 2.7: G_t+1 = sigmoid(sum w * agg(x) - theta)
    Broadcasting mechanism.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Aggregation weights: Maps module outputs to workspace "bus"
        # We assume simplified "Perception" and "Spiking Core" provide inputs.
        # Actually, let's just take the Embedding and the Spike State.
        self.w_enc = nn.Linear(embed_dim, embed_dim)
        self.w_state = nn.Linear(Config.HIDDEN_DIM, embed_dim)
        
        # The Ignition Gate
        # G is a scalar or small vector indicating relevance/attention?
        # Prompt: "G_t ... Broadcast ... x_new = x + w G"
        # Let's make G same dim as embedding for rich broadcast.
        self.ignition_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )
        
        # Feedback weights: G -> Module Input
        self.w_feedback = nn.Linear(embed_dim, embed_dim) # Feedback to Encoder/LIF input
        
        self.threshold = Config.G_THRESHOLD

    def forward(self, x_enc, s_state):
        """
        x_enc: (Batch, Embed)
        s_state: (Batch, Hidden)
        """
        # Aggregation
        # Sum inputs from modules
        total_activation = self.w_enc(x_enc) + self.w_state(s_state)
        
        # Ignition (Soft)
        # G_t in [0, 1]
        g_raw = self.ignition_gate(total_activation)
        
        # Thresholding / Hard Ignition check (for monitoring)
        ignition_event = (g_raw.mean() > self.threshold).float()
        
        # The actual value used for broadcast is g_raw (soft) or Gated?
        # Math says: G = sigmoid( ... - theta). 
        # If we learned bias in the Linear layer, that acts as theta.
        
        # Broadcast signal
        broadcast = self.w_feedback(g_raw)
        
        return broadcast, ignition_event

import torch
import torch.nn as nn
from config import Config

class LIFLayer(nn.Module):
    """
    2.1 Discrete-Time LIF Neuron Dynamics
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Synaptic weights: W_ij
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        
        # State
        self.v = None   # Membrane potential
        self.s = None   # Spike state
        
    def reset_state(self, batch_size):
        """Initialize voltage and spikes."""
        self.v = torch.zeros(batch_size, self.output_dim, device=Config.DEVICE)
        self.s = torch.zeros(batch_size, self.output_dim, device=Config.DEVICE)
        
    def forward(self, x_in):
        """
        Input:
            x_in: Inputs from previous layer or encoder x_j(t)
        Output:
            s_out: Spikes s_i(t)
            v_out: Membrane potential (for monitoring)
        """
        # Eq 2.1: Membrane update
        # Current input: I_i(t) = sum_j w_ij * x_j(t) + b_i
        current = self.fc(x_in)
        
        # v_i(t+1) = alpha * v_i(t) + (1-alpha) * current
        # Note: We apply reset from PREVIOUS step before integrating new input if strictly following some conventions,
        # but the prompt specifies: v(t+1) = reset IF s(t)=1.
        # This implies we check spike, THEN update voltage for next step or reset.
        
        # Let's follow a standard discrete update:
        # 1. Integrate: v_temp = alpha * v_old + (1-alpha) * input
        # 2. Spike: s_new = 1 if v_temp >= Vth
        # 3. Reset: v_new = v_reset if s_new else v_temp
        
        # The prompt Eq is:
        # v(t+1) = alpha * v(t) + ... 
        # s(t) = I{v(t+1) >= Vth}
        # v(t+1) <- reset if s(t)=1 (effectively for next step usage)
        
        # Using variable names for t+1
        v_next = Config.ALPHA * self.v + (1.0 - Config.ALPHA) * current
        
        # Spike generation
        # Heaviside step function
        s_next = (v_next >= Config.V_TH).float()
        
        # Reset rule (Soft reset or Hard reset? Prompt says v_reset)
        # v_i(t+1) = v_reset if s_i(t)=1
        # This is effectively applied for the START of t+2. 
        # But we return the v_next as the state at t+1.
        # We must store the reset version for the next iteration.
        
        v_final = s_next * Config.V_RESET + (1 - s_next) * v_next
        
        # Update internal state
        self.v = v_final.detach() # Detach to stop BPTT through time if we aren't using BPTT (we use e-traces)
        self.s = s_next.detach()
        
        return s_next, v_final

import torch
from config import Config

class SynapseManager:
    """
    Handles Plasticity & Neuromodulation
    Eq 2.2: Eligibility Trace Dynamics
    Eq 2.4: Neuromodulated Synaptic Weight Update
    """
    def __init__(self, layer, input_dim, output_dim):
        self.layer = layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Eligibility Traces e_ij
        self.traces_w = torch.zeros(output_dim, input_dim, device=Config.DEVICE)
        self.traces_b = torch.zeros(output_dim, device=Config.DEVICE)
        
    def reset_traces(self):
        self.traces_w.zero_()
        self.traces_b.zero_()
        
    def update_traces(self, x_in, s_out):
        """
        Eq 2.2: e_ij(t+1) = lambda_e * e_ij(t) + s_i(t) * x_j(t)
        
        x_in shape: (Batch, Input) -> x_j
        s_out shape: (Batch, Output) -> s_i
        We average over batch for batch-mode updates.
        """
        batch_size = x_in.shape[0]
        
        # Outer product for Hebbian term: s_i * x_j
        # s_out.T @ x_in -> (Output, Batch) @ (Batch, Input) -> (Output, Input)
        hebbian_w = (s_out.T @ x_in) / batch_size
        hebbian_b = s_out.mean(dim=0) # Bias trace just driven by output spike
        
        self.traces_w = Config.LAMBDA_E * self.traces_w + hebbian_w
        self.traces_b = Config.LAMBDA_E * self.traces_b + hebbian_b
        
    def apply_weight_update(self, td_error):
        """
        Eq 2.4: Delta w_ij(t) = eta_w * delta_t * e_ij(t) - lambda_w * w_ij(t)
        """
        # Ensure td_error is a scalar or matches dimensions
        if isinstance(td_error, torch.Tensor):
            delta = td_error.item()
        else:
            delta = td_error
            
        with torch.no_grad():
            # Update weights
            eff_trace_w = delta * self.traces_w
            decay_w = Config.LAMBDA_W * self.layer.fc.weight.data
            
            d_weight = Config.ETA_W * eff_trace_w - decay_w
            self.layer.fc.weight.data += d_weight
            
            # Update biases
            eff_trace_b = delta * self.traces_b
            decay_b = Config.LAMBDA_W * self.layer.fc.bias.data
            
            d_bias = Config.ETA_W * eff_trace_b - decay_b
            self.layer.fc.bias.data += d_bias

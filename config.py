import torch

class Config:
    # --------------------------
    # 2.1 LIF Neuron Parameters
    # --------------------------
    DT = 1.0                # Time step (ms)
    TAU_M = 20.0            # Membrane time constant (ms)
    ALPHA = torch.exp(torch.tensor(-DT / TAU_M)) # Decay factor (Eq 2.1)
    
    V_TH = 1.0              # Spike threshold
    V_RESET = 0.0           # Reset potential
    
    # --------------------------
    # 2.2 Plasticity Parameters
    # --------------------------
    LAMBDA_E = 0.95         # Eligibility trace decay (Eq 2.2)
    LAMBDA_W = 1e-4         # Weight decay
    ETA_W = 0.01            # Learning rate for plasticity (Eq 2.4)
    
    # --------------------------
    # 2.3 RL Parameters
    # --------------------------
    GAMMA = 0.99            # Discount factor
    
    # --------------------------
    # Optimization Parameters
    # --------------------------
    LR_PRED = 0.05          # Predictor learning rate (Aggressive)
    LR_ACTOR_CRITIC = 0.05  # Actor-Critic learning rate
    LAMBDA_PRED = 1.0       # Loss weight: Prediction
    LAMBDA_V = 1.0          # Loss weight: Value
    LAMBDA_PI = 1.0         # Loss weight: Policy
    LAMBDA_REG = 1e-5       # Loss weight: Regularization
    
    # --------------------------
    # Architecture
    # --------------------------
    INPUT_DIM = 10          # Vocab Size (Small)
    HIDDEN_DIM = 64         # Number of spiking neurons
    EMBED_DIM = 32          # Dimension of sensory embedding
    ACTION_DIM = 10         # Action matches Vocab
    
    # --------------------------
    # Global Workspace Params
    # --------------------------
    G_THRESHOLD = 0.5       # Ignition threshold
    
    # --------------------------
    # Transformer Params
    # --------------------------
    TRANSFORMER_HEADS = 2
    TRANSFORMER_LAYERS = 1
    
    # --------------------------
    # Simulation
    # --------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = 10

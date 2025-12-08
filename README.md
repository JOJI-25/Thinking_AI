# Biologically Inspired Cognitive AI

This project implements a biologically plausible Artificial Intelligence system based on Computational Neuroscience principles. It goes beyond standard Reinforcement Learning by incorporating spiking neurons, a global workspace for working memory, and a "System 2" deliberation loop.

## Key Features

1.  **Spiking Neural Networks (SNN)**: Uses Leaky Integrate-and-Fire (LIF) neurons (`neuron.py`) with continuous dynamics.
2.  **Neuromodulated Plasticity**: Learns via biologically grounded rules (Eligibility Traces + TD Error) (`plasticity.py`) instead of standard Backpropagation-through-time.
3.  **Global Workspace Theory (GWT)**: Implements a "Conscious" workspace that broadcasts important signals to all modules (`workspace.py`), enabling Working Memory.
4.  **World Model**: A Transformer-based predictor (`modules.py`) that allows the agent to simulate futures.
5.  **System 2 Reasoning**: The agent can pause to "think" (`deliberate_action` in `agent.py`), simulating various futures before acting.
6.  **Generative Modeling**: Capable of learning sequential data (Text) as a Spiking Language Model (`train_generative.py`).

## File Structure

- `agent.py`: The main `BioAgent` class integrating all brain modules.
- `thinking.py`: The main Reinforcement Learning loop (GridWorld).
- `train_generative.py`: Training script for Text Generation (LLM mode).
- `test_agent.py`: Visualization script to watch the agent "think" and solve tasks.
- `config.py`: Hyperparameters for neurons, plasticity, and dimensions.
- `environment.py`: Custom GridWorld environments (including Memory tasks).
- `text_environment.py`: Streaming text environment for generative training.
- `neuron.py`, `plasticity.py`: Core biological mechanism implementations.

## Usage

### 1. Train the Agent (Navigation Task)
```bash
python think.py
```

### 2. Watch the Agent Think (System 2 Demo)
```bash
python test_agent.py
```
*Output will show the agent's internal monologue.*

### 3. Train a Biological LLM
```bash
python train_generative.py
```

## Theory

This system allows us to move from "Function Approximators" (Standard AI) to "Cognitive Agents" that perceive, attend, remember, and deliberate.

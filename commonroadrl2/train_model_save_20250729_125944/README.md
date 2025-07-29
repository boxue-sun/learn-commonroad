# Current Model Save (62,100 Episodes)

Saved on: 2025-07-29 12:59:44

## Model Information
- Model: q_network_62100_episodes.pth
- Training Episodes: 62,100
- Target Episodes: 100,000
- Completion: 62.1%
- State Dimension: 33
- Action Dimension: 25
- Hidden Dimension: 128

## Training Status
- Status: Interrupted at 62,100 episodes
- Model: Q-Network with 4 layers
- Architecture: 33 -> 128 -> 128 -> 128 -> 25

## Usage
To load this model:
```python
import torch
from train_multi_scenario import QNetwork

model = QNetwork(state_dim=33, action_dim=25)
model.load_state_dict(torch.load('q_network_62100_episodes.pth'))
```

## Note
This model represents the state at 62,100 training episodes.
The training was interrupted before completion.

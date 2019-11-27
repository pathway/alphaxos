# AlphaXos

## Project status
- experimental!

### Self play with Deep Reinforcement Learning: Deep Q-Learning using board games in an Open AI Gym-like Environment

What?
- Concise working example of self-play Deep Q-Learning
- You may find it a useful example in contrast to Alpha Zero / AlphaGo Zero (AZ/AGZ)
- Environment similar to those in OpenAI gym at gym/envs/toy_text/ 
- General approach to piece-placement board games

Agents:
- ChaosAgent: Same as DQNAgent, but Epsilon-greedy during play (not just during training)
- DQNAgent: Double-Deep Q-Learning agent trained with keras-rl
- RandomAgent: always plays a random (but valid) move
- HumanAgent: takes keyboard input

### Comparison with AlphaZero / AlphaGo Zero

There is no tree search here, its just one-step DQN for now.  So a completely different type of RL (model-free) than AlphaZero (model-based with tree search).

Similar to AZ/AGZ:
- reinforcement learning for a binary board game
- game state represented via board input matrix
- uses single neural network (aside from the fact it uses double DQN), instead of separate policy and value networks like earlier AlphaGos
- learns entirely from self-play (in the case of AlphaXos, also learns from play against purely random player, as well as self-play)
- no human-engineered features or logic

Different from AZ/AGZ:
- AX uses Double Deep Q Learning (via keras-rl), as opposed to the novel Monte Carlo Tree Search variation of Policy Improvement used by AZ/AGZ, which I think was the meat of their contribution
- AGZ used rotated/reflected board positions to increase sample efficiency.  AZ did not do this.  AlphaXos does not currently do this.
- uses a simple shallow keras FF network (instead of a deep residual convolutional network in the case of AGZ)
- uses single 2D matrix for representing board including both players, instead of a multi-layer matrix like AZ/AGZ.  The games we consider here do not require previous timesteps in order to completely capture game state.  Ie. here the current board state is sufficient to satisfy the Markhov assumption for an MDP.
- adjusts representation of board depending on turn side, as opposed to AGZ which provides turn side as input to the network
- probably many other things!

### Next steps

- lots

### References

- Alpha Zero: https://arxiv.org/abs/1712.01815
- AlphaGo Zero: https://deepmind.com/documents/119/agz_unformatted_nature.pdf
- OpenAI gym: https://github.com/openai/gym/
- Keras-RL: https://github.com/keras-rl/keras-rl
- Keras: https://keras.io/
- DQNs: https://www.nature.com/articles/nature14236
- Double DQN: https://arxiv.org/abs/1509.06461

Copyright (c) 2018 Robin Chauhan

License: The MIT License

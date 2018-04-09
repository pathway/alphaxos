# AlphaXos

### Self play with Deep Reinforcement Learning: Deep Q-Learning using Xs and Os in an Open AI Gym-like Environment

What?
- Concise working example of self-play Deep Q-Learning
- You may find it a useful example in discussing and understanding AlphaGo Zero / AGZ
- Environment similar to those in OpenAI gym at gym/envs/toy_text/ 

Similar to AlphaGo Zero:
- reinforcement learning for a binary board game
- game state represented via input matrix
- uses single neural network, instead of separate policy and value networks like earlier AlphaGos
- learns entirely from self-play (in the case of AlphaXos, also learns from play against purely random player)
- no human-engineered features or logic

Different from AlphaGo Zero:
- uses Deep Q Learning
- uses a shallow keras FF network (instead of a deep residual network in the case of AGZ)
- uses single matrix for representing board instead of a multi-layer matrix like AGZ
- adjusts representation of board depending on turn side, as opposed to AlphaGo Zero which inputs side to the network
- does not perform any Monte-Carlo Tree Search, 
- probably 

Agents:
- ChaosAgent: Epsilon-greedy during play (not just during training)
- DQNAgent: Double-Deep Q-Learning agent trained with keras-rl
- RandomAgent: always plays a random (but valid) move
- HumanAgent: takes keyboard input

Next steps:
- lots

### References

- AlphaGo Zero: https://deepmind.com/documents/119/agz_unformatted_nature.pdf
- OpenAI gym: https://github.com/openai/gym/
- Keras-RL: https://github.com/keras-rl/keras-rl
- Keras: https://keras.io/
- DQNs: https://www.nature.com/articles/nature14236

Copyright (c) 2018 Robin Chauhan

License: The MIT License

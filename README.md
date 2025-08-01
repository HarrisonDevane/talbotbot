# Talbotbot 😈

**Version:** 1.5  
**Approximate Rating:** 1700 Lichess bot elo (human equivalent: ~1800–1900)  
**Type:** Chess AI combining Monte Carlo Tree Search (MCTS) with a Convolutional Neural Network (CNN)

Talbotbot is a chess engine that uses a deep neural network to evaluate positions and guide move selection with MCTS.

## Demo

Play against Talbotbot or watch it on Lichess:  
https://lichess.org/@/Talbotbot

## Project Structure

```
Talbotbot/
├── data/                             # Links to raw position data, engine data and assets
│ 
├── inference/                        # Inference and gameplay components
│   ├── agents/                       # Agents implementing playing logic
│   │   └── talbot.py                 # High-level player agent using MCTS + NN
│   ├── mcts/                         # Monte Carlo tree search components
│   │   ├── mcts_engine.py            # Main MCTS script called by player instance
│   │   └── mcts_node.py              # Node for the MCTS
│   └── scripts/                      # Utility scripts for running inference
│       ├── lichess_inference.py      # Lichess API integration for live play
│       └── local_inference.py        # Local gameplay and testing
│
├── training
│   └── supervised/                   # Supervised learning runs by version (v1, v2, etc.)
│
├── model.py                          # Model shared for both inference and training
└── utils.py                          # Utility script shared for both inference and training
```

## Model Architecture

Talbotbot’s core is a deep residual CNN inspired by AlphaZero, designed to predict move probabilities and the expected outcome of a position.

- **Input:** 68 feature planes representing the board state and recent move history  
- **Network:**  
  - Initial convolution layer  
  - 20 residual blocks with batch normalization and ReLU activations  
  - Dropout applied from block 10 onwards to improve generalization  
- **Outputs:**  
  - **Policy head:** predicts probabilities for 4672 possible moves  
  - **Value head:** predicts the position’s expected outcome as a number between −1 (loss) and +1 (win)

## Monte Carlo Tree Search (MCTS)

Talbotbot uses MCTS to explore possible moves by simulating games and using the neural network’s policy and value predictions to guide search efficiently. This helps balance exploration of new moves with exploitation of strong known moves, improving decision-making during play.

## Training Data

Talbotbot v1.5 was trained with supervised learning on diverse datasets, including:

- 20 million positions from Grandmaster games  
- 5 million positions from Lichess games rated 2600+  
- 2 million tactical positions  
- 2 million synthetic endgame positions (random legal setups with 3–7 pieces)

Positions are labeled using quick Stockfish evaluations (~0.02 seconds per position) and corresponding moves from games or engine analysis. For snapshot positions (tactics, endgames), prior states are reconstructed by playing moves backward.
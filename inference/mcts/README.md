# Monte Carlo Tree Search (MCTS) Engine

This module implements the Monte Carlo Tree Search (MCTS) algorithm used by Talbotbot to select strong moves in a given chess position. It combines search-based decision-making with neural network evaluations for both move probabilities (policy) and board evaluation (value), inspired by AlphaZero-style architectures.

## Overview

MCTS is a heuristic search algorithm for decision processes, particularly effective in games like chess. Talbotbot's MCTS performs simulations of possible future moves, guided by neural network predictions, and uses the results to make principled decisions under uncertainty.

Each simulation follows four key steps:

1. **Selection:** Traverse the tree using the UCT (Upper Confidence Bound applied to Trees) score to choose promising child nodes.
2. **Expansion:** If a leaf node is reached, it is expanded by generating all legal child moves.
3. **Evaluation:** The neural network evaluates the new node, providing a policy distribution (over all legal moves) and a scalar value (indicating win probability).
4. **Backpropagation:** The result of the evaluation is backpropagated up the tree to update visit counts and value estimates.

## Key Features

- **Neural-Guided Search:** The engine uses a policy/value network to focus simulations on strong moves.
- **Batch Inference:** Nodes awaiting evaluation are queued and processed in batches for efficient GPU usage.
- **Tree Reuse:** When possible, the search tree is reused across moves to preserve previously gathered information.
- **Game-Aware Resetting:** If moves diverge from expected paths, the tree resets appropriately.
- **Cython-Accelerated Move Generation:** Fast legal move enumeration via `cython_chess`.

## Usage

The MCTS engine is driven through the `MCTSEngine` class and requires:
- A reference to a neural network policy/value model (`model_player`)
- A board state (`chess.Board`)
- A time budget for running simulations

Tree traversal, expansion, and inference are managed internally. The most visited child of the root node after simulations is typically chosen as the best move.
# Talbotbot v1.1

## Overview

This version represents the initial proof of concept: a small-scale model trained on a limited dataset, without MCTS integration. The model has the following characteristics:

- **Network:** 10 residual blocks  
- **Training data:** 5 million Lichess positions from the Elite Database  
- **Input:** A tensor of shape **18 × 8 × 8** representing the board state, composed of the following feature planes:
    - 6 planes for white pieces: One binary plane per piece type (pawn, knight, bishop, rook, queen, king); `1` indicates presence of the piece, `0` indicates absence.
    - 6 planes for black pieces: Same format as above, for black's pieces.
    - 1 plane for the current turn: All `1`s if it's white to move, all `0`s if it's black.
    - 4 planes for castling rights: Kingside and queenside availability for both white and black; each plane is filled with `1`s if castling is available, otherwise `0`s.
    - 1 plane for en passant: A binary mask where only the valid en passant file is marked with `1`s (if applicable).
- **Policy head:** Predicts a one-hot encoded vector for the move that was played.
- **Value head:** Predicts the game outcome (win/loss/draw) as `1`, `0`, or `-1` from the perspective of the current player.

## Training

The following hyperparameters were used for this iteration. No regularization techniques were applied:

### Hyperparameters

- **Batch size:** 512  
- **Learning rate:** 1e-3  
- **Scheduler:** ReduceLROnPlateau  
- **Optimizer:** Adam  
- **Training set size:** 2%

## Training loss

Below is the plot showing total training loss across epochs:

![Training Loss](logs/training_loss.png)

- **V_Loss:** The loss from the value head, measuring mean squared error in predicting the final game outcome (a scalar between -1 and 1). Since this is a single continuous target, the loss tends to be relatively small.
- **P_Loss:** The policy head loss, computed using cross-entropy between the predicted move distribution and the one-hot encoded actual move. This loss is typically higher due to the large move space.
- **T_Loss:** The total training loss, calculated as the sum of V_Loss and P_Loss. This combined loss is used to track model performance and determine the best checkpoint.

The following plot compares the average training loss per epoch with the corresponding validation loss:

![Training vs Validation Loss](logs/training_vs_validation_loss.png)

Due to the small model size, limited dataset, and lack of regularization, the model begins to overfit quickly. The best performance occurs around epoch 10.

## Evaluation

- No search algorithm was used; the move with the highest probability from the policy head was selected directly.
- The model performed reasonably well in the opening phase but struggled with short tactical sequences.
- The value head exhibited significant variance, which is expected given it was trained to predict the final game result rather than a positional evaluation.

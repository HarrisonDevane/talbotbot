# Talbotbot v1.1

## Overview

The proof of concept version. This was a small scale model training on a small dataset with no MCTS integration. This model had the following properties:

- **Network:**  10 residual blocks
- **Data:** 5 million lichess positions from the elite DB. Positions were stored in hdf5 file containing:
  - **Input:** 18 feature planes representing the board state
  - **Policy head:** one hot encoded move that was played
  - **Value head:** game outcome (win/loss/draw) as 1, 0, -1 from perspective of player

## Training

The hyper paramters used for this iteration are below. No regularization was used

### Hyperparameters
- **Batch size:**  512
- **Learning rate:**  1e-3
- **Scheduler:**  ReduceLROnPlateau
- **Optimizer:**  Adam
- **Training set size:**  2%

## Training loss

Here is the plot showing the training loss per epoch:

![Training Loss](log/training_loss.png)

- **V_Loss:**  Value head loss, giving essentially the probability of winning. As this is a single continuous variable [-1,1], the loss is smaller
- **P_Loss:**  Policy head loss, giving essentially the probability of output moves. Much higher as this is comparing a one-hot encoded value with a distribution
- **T_Loss:**  Sum of V_Loss and P_Loss, used for determining best model

Here is the plot comparing the average training loss per epoch, with the average validation loss per epoch:

![Training vs Validation Loss](log/training_vs_validation.png)

Due to the small model size, relatively small dataset, and no regularization, we see rapid overfitting. Best model is ~epoch 10.

## Evaluation

- Did not include search algorithm, meaning the highest probability move from the policy head is the one played.
- Played well in opening, but failed to see short tactical lines.
- Value head varies wildly. Makes sense, as we trained on the outcome of the game as opposed to present valuation.
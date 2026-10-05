from pathlib import Path

import torch


class EarlyStopping:
    """
    Stop training when validation loss does not improve
    for a specified number of epochs.
    """

    def __init__(
        self,
        patience=7,
        min_delta=0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        val_loss,
    ):
        improved = (
            val_loss
            < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = val_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return improved


def save_checkpoint(
    model,
    path,
):
    """
    Save model state_dict.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )
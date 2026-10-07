from pathlib import Path

import pandas as pd
import torch

from src.training.callbacks import (
    EarlyStopping,
    save_checkpoint,
)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    print_freq: int = 50,
):
    """
    Train model for one epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, 1):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        if print_freq > 0 and (batch_idx % print_freq == 0 or batch_idx == total_batches):
            print(
                f"  [Batch {batch_idx:3d}/{total_batches}] "
                f"Loss: {running_loss / total:.4f} | "
                f"Acc: {correct / total:.4f} ({correct / total * 100:.1f}%)",
                flush=True,
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


def evaluate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate model for one epoch.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    checkpoint_path,
    scheduler=None,
    early_stopping_patience=7,
    history_path=None,
):
    """
    Generic model training loop.

    Can be reused for:
        - Simple CNN
        - Complex CNN
        - Transfer Learning
        - Fine-Tuning
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    early_stopping = EarlyStopping(
        patience=early_stopping_patience
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    for epoch in range(
        1,
        num_epochs + 1,
    ):
        print(f"\n--- Epoch {epoch:02d}/{num_epochs} ---", flush=True)

        train_loss, train_acc = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_acc = (
            evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_acc
        )

        history["learning_rate"].append(
            current_lr
        )

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        improved = early_stopping.step(
            val_loss
        )

        if improved:

            save_checkpoint(
                model=model,
                path=checkpoint_path,
            )

            print(
                "  -> Best model saved.",
                flush=True,
            )

        if scheduler is not None:

            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(
                    val_loss
                )

            else:
                scheduler.step()

        if history_path is not None:

            history_df = pd.DataFrame(
                history
            )

            history_path = Path(
                history_path
            )

            history_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            history_df.to_csv(
                history_path,
                index=False,
            )

        if early_stopping.should_stop:

            print(
                "Early stopping triggered."
            )

            break

    return pd.DataFrame(
        history
    )
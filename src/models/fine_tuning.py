import torch
import torch.nn as nn


def configure_resnet18_fine_tuning(model):
    """
    Configure progressive fine-tuning for TransferLearningResNet18.

    Strategy:
    - Freeze all model parameters.
    - Unfreeze ResNet18 layer4.
    - Unfreeze the task-specific classifier.

    Parameters
    ----------
    model:
        TransferLearningResNet18 model.

    Returns
    -------
    model
        Model with appropriate requires_grad configuration.
    """

    # Freeze everything first
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final residual stage
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad = True

    # Keep classifier trainable
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad = True

    return model


def get_fine_tuning_criterion(
    raw_class_weights,
    label_smoothing=0.05,
):
    """
    Create the loss function used during fine-tuning.

    The original inverse-frequency class weights may be very large
    for rare HAM10000 classes. Taking the square root softens the
    imbalance correction while still giving minority classes more
    importance than the majority class.

    Parameters
    ----------
    raw_class_weights:
        Class weights calculated from the training split.

    label_smoothing:
        Label smoothing factor.

    Returns
    -------
    criterion:
        Weighted CrossEntropyLoss.

    fine_tune_class_weights:
        Softened class weights.
    """

    fine_tune_class_weights = torch.sqrt(
        raw_class_weights
    )

    criterion = nn.CrossEntropyLoss(
        weight=fine_tune_class_weights,
        label_smoothing=label_smoothing,
    )

    return criterion, fine_tune_class_weights


def get_fine_tuning_optimizer(
    model,
    backbone_lr=1e-5,
    classifier_lr=1e-4,
    weight_decay=1e-4,
):
    """
    Create AdamW optimizer with discriminative learning rates.

    layer4 receives a smaller learning rate because it contains
    pretrained ImageNet features.

    The classifier receives a larger learning rate because it needs
    stronger task-specific adaptation.

    Parameters
    ----------
    model:
        Fine-tuning model.

    backbone_lr:
        Learning rate for ResNet18 layer4.

    classifier_lr:
        Learning rate for the classification head.

    weight_decay:
        AdamW weight decay.

    Returns
    -------
    optimizer
        Configured AdamW optimizer.
    """

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.layer4.parameters(),
                "lr": backbone_lr,
            },
            {
                "params": model.backbone.fc.parameters(),
                "lr": classifier_lr,
            },
        ],
        weight_decay=weight_decay,
    )

    return optimizer


def get_fine_tuning_scheduler(
    optimizer,
    factor=0.5,
    patience=2,
    min_lr=1e-7,
):
    """
    Create ReduceLROnPlateau scheduler.

    Learning rate is reduced when validation loss stops improving.

    Parameters
    ----------
    optimizer:
        PyTorch optimizer.

    factor:
        Multiplicative learning-rate reduction factor.

    patience:
        Number of epochs without improvement before reducing LR.

    min_lr:
        Minimum learning rate.

    Returns
    -------
    scheduler
        ReduceLROnPlateau scheduler.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

    return scheduler


def count_trainable_parameters(model):
    """
    Count total, trainable, and frozen model parameters.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    dict
        Parameter statistics.
    """

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_params = (
        total_params
        - trainable_params
    )

    trainable_ratio = (
        trainable_params
        / total_params
    )

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_ratio": trainable_ratio,
    }
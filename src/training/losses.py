import torch
import torch.nn as nn


def compute_class_weights(
    train_df,
    class_names,
    device=None,
):
    """
    Compute balanced class weights:

        weight_c = N / (K * n_c)

    where:
        N   = total number of training samples
        K   = number of classes
        n_c = number of samples in class c
    """

    class_counts = (
        train_df["dx"]
        .value_counts()
        .reindex(class_names)
    )

    if class_counts.isna().any():
        raise ValueError(
            "Some classes are missing from the training split."
        )

    num_samples = len(train_df)
    num_classes = len(class_names)

    weights = []

    for class_name in class_names:
        count = class_counts[class_name]

        weight = (
            num_samples
            / (num_classes * count)
        )

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    if device is not None:
        weights = weights.to(device)

    return weights


def get_weighted_cross_entropy_loss(
    class_weights,
):
    """
    Weighted CrossEntropyLoss for imbalanced classification.
    """

    return nn.CrossEntropyLoss(
        weight=class_weights
    )
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RANDOM_STATE = 42


def validate_split_columns(df: pd.DataFrame) -> None:
    """
    Validate columns required for lesion-level stratified splitting.
    """
    required_columns = {
        "lesion_id",
        "image_id",
        "dx",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def check_lesion_label_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether each lesion_id belongs to exactly one diagnosis class.

    Returns a DataFrame containing inconsistent lesions.
    Empty DataFrame means all lesions are consistent.
    """
    lesion_class_counts = (
        df.groupby("lesion_id")["dx"]
        .nunique()
        .reset_index(name="num_classes")
    )

    inconsistent_lesions = lesion_class_counts[
        lesion_class_counts["num_classes"] > 1
    ].copy()

    return inconsistent_lesions


def build_lesion_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per lesion.

    Each lesion receives its diagnosis label and number of images.
    """
    validate_split_columns(df)

    inconsistent = check_lesion_label_consistency(df)

    if not inconsistent.empty:
        raise ValueError(
            "Some lesion_id values are associated with multiple "
            "diagnosis labels. Resolve them before splitting.\n"
            f"Inconsistent lesions:\n{inconsistent.head()}"
        )

    lesion_table = (
        df.groupby("lesion_id")
        .agg(
            dx=("dx", "first"),
            num_images=("image_id", "count"),
        )
        .reset_index()
    )

    return lesion_table


def stratified_lesion_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform lesion-level stratified Train / Validation / Test split.

    Important:
    - Splitting is performed on lesion_id, NOT individual images.
    - All images from the same lesion remain in the same split.
    - Stratification is performed using diagnosis class (dx).

    Parameters
    ----------
    df:
        Cleaned HAM10000 metadata.

    train_size:
        Fraction assigned to training set.

    val_size:
        Fraction assigned to validation set.

    test_size:
        Fraction assigned to test set.

    random_state:
        Seed for reproducibility.

    Returns
    -------
    train_df, val_df, test_df
    """

    validate_split_columns(df)

    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0"
        )

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "All split sizes must be greater than 0."
        )

    # ---------------------------------------------------------
    # Step 1: create lesion-level table
    # ---------------------------------------------------------

    lesion_table = build_lesion_table(df)

    # ---------------------------------------------------------
    # Step 2: Train vs Temporary
    # ---------------------------------------------------------

    temp_size = val_size + test_size

    train_lesions, temp_lesions = train_test_split(
        lesion_table,
        test_size=temp_size,
        random_state=random_state,
        stratify=lesion_table["dx"],
    )

    # ---------------------------------------------------------
    # Step 3: Validation vs Test
    # ---------------------------------------------------------

    relative_test_size = test_size / temp_size

    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_lesions["dx"],
    )

    # ---------------------------------------------------------
    # Step 4: map lesion split back to images
    # ---------------------------------------------------------

    train_ids = set(train_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    train_df = (
        df[df["lesion_id"].isin(train_ids)]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        df[df["lesion_id"].isin(val_ids)]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[df["lesion_id"].isin(test_ids)]
        .copy()
        .reset_index(drop=True)
    )

    # Add explicit split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def check_lesion_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, set]:
    """
    Check whether any lesion_id appears in more than one split.
    """

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    leakage = {
        "train_val": train_lesions & val_lesions,
        "train_test": train_lesions & test_lesions,
        "val_test": val_lesions & test_lesions,
    }

    return leakage


def check_image_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, set]:
    """
    Check whether any image_id appears in more than one split.
    """

    train_images = set(train_df["image_id"])
    val_images = set(val_df["image_id"])
    test_images = set(test_df["image_id"])

    leakage = {
        "train_val": train_images & val_images,
        "train_test": train_images & test_images,
        "val_test": val_images & test_images,
    }

    return leakage


def validate_split_integrity(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Perform integrity checks after splitting.
    """

    lesion_leakage = check_lesion_leakage(
        train_df,
        val_df,
        test_df,
    )

    image_leakage = check_image_leakage(
        train_df,
        val_df,
        test_df,
    )

    total_after_split = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    all_lesion_leakage_empty = all(
        len(values) == 0
        for values in lesion_leakage.values()
    )

    all_image_leakage_empty = all(
        len(values) == 0
        for values in image_leakage.values()
    )

    all_images_preserved = (
        total_after_split == len(original_df)
    )

    original_image_ids = set(original_df["image_id"])

    split_image_ids = (
        set(train_df["image_id"])
        | set(val_df["image_id"])
        | set(test_df["image_id"])
    )

    same_image_set = (
        original_image_ids == split_image_ids
    )

    report = {
        "original_images": len(original_df),
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "total_after_split": total_after_split,
        "all_images_preserved": all_images_preserved,
        "same_image_set": same_image_set,
        "lesion_leakage": lesion_leakage,
        "image_leakage": image_leakage,
        "no_lesion_leakage": all_lesion_leakage_empty,
        "no_image_leakage": all_image_leakage_empty,
    }

    return report


def get_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create high-level summary of split sizes.
    """

    total_images = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    rows = []

    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:

        rows.append({
            "split": name,
            "num_images": len(split_df),
            "num_lesions": split_df["lesion_id"].nunique(),
            "percentage_images": (
                len(split_df) / total_images * 100
            ),
        })

    return pd.DataFrame(rows)


def get_class_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return class count and percentage for a split.
    """

    counts = (
        df["dx"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df["dx"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    })

    result.index.name = "dx"

    return result


def combine_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine train, validation and test metadata.
    """

    combined_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    return combined_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save individual and combined split metadata.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df.to_csv(
        output_dir / "val.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    combined_df = combine_splits(
        train_df,
        val_df,
        test_df,
    )

    combined_df.to_csv(
        output_dir / "HAM10000_metadata_split.csv",
        index=False,
    )
from pathlib import Path
import pandas as pd
from PIL import Image


VALID_CLASSES = {
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
}


def collect_image_paths(raw_data_dir: Path):
    """
    Collect all JPG/JPEG image files recursively
    and return a mapping:
        image_id -> image_path
    """
    image_paths = list(raw_data_dir.rglob("*.jpg"))
    image_paths += list(raw_data_dir.rglob("*.jpeg"))

    image_map = {
        path.stem: path
        for path in image_paths
    }

    return image_map


def remove_exact_duplicates(df: pd.DataFrame):
    """
    Remove rows that are exact duplicates.
    """
    before = len(df)

    df = df.drop_duplicates().copy()

    removed = before - len(df)

    return df, removed


def remove_duplicate_image_ids(df: pd.DataFrame):
    """
    Ensure each image_id appears only once.
    """
    before = len(df)

    df = (
        df
        .drop_duplicates(subset=["image_id"], keep="first")
        .copy()
    )

    removed = before - len(df)

    return df, removed


def filter_valid_classes(df: pd.DataFrame):
    """
    Keep only the 7 expected HAM10000 diagnosis classes.
    """
    before = len(df)

    df = df[df["dx"].isin(VALID_CLASSES)].copy()

    removed = before - len(df)

    return df, removed


def attach_image_paths(
    df: pd.DataFrame,
    image_map: dict
):
    """
    Add image_path column based on image_id.
    """
    df = df.copy()

    df["image_path"] = df["image_id"].map(image_map)

    return df


def remove_missing_image_files(df: pd.DataFrame):
    """
    Remove metadata rows whose image file cannot be found.
    """
    before = len(df)

    df = df[df["image_path"].notna()].copy()

    removed = before - len(df)

    return df, removed


def validate_images(df: pd.DataFrame):
    """
    Detect unreadable/corrupted image files.
    Returns:
        valid_df
        corrupted_records
    """
    valid_indices = []
    corrupted_records = []

    for idx, row in df.iterrows():
        image_path = Path(row["image_path"])

        try:
            with Image.open(image_path) as img:
                img.verify()

            valid_indices.append(idx)

        except Exception as exc:
            corrupted_records.append({
                "image_id": row["image_id"],
                "image_path": str(image_path),
                "error": str(exc),
            })

    valid_df = df.loc[valid_indices].copy()

    return valid_df, corrupted_records


def clean_metadata_strings(df: pd.DataFrame):
    """
    Normalize metadata string columns.

    Identifier columns such as image_id and lesion_id
    preserve their original case.

    Categorical columns are normalized to lowercase.
    """
    df = df.copy()

    # IDs: strip whitespace only, DO NOT lowercase
    id_columns = [
        "lesion_id",
        "image_id",
    ]

    for column in id_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    # Categorical fields: lowercase for consistency
    categorical_columns = [
        "dx",
        "dx_type",
        "sex",
        "localization",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return df


def clean_age(df: pd.DataFrame):
    """
    Convert age to numeric and invalidate impossible values.

    Missing ages are kept as NaN because age is not
    currently used as a CNN input feature.
    """
    df = df.copy()

    df["age"] = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    df.loc[
        (df["age"] < 0) | (df["age"] > 120),
        "age"
    ] = pd.NA

    return df


def validate_required_columns(df: pd.DataFrame):
    """
    Check that essential HAM10000 columns exist.
    """
    required_columns = {
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def clean_ham10000(
    df: pd.DataFrame,
    raw_data_dir: Path,
    verify_images: bool = True
):
    """
    Complete metadata cleaning pipeline.

    Returns:
        cleaned_df
        report
    """
    validate_required_columns(df)

    report = {
        "initial_rows": len(df)
    }

    # 1. Normalize metadata strings
    df = clean_metadata_strings(df)

    # 2. Clean age values
    df = clean_age(df)

    # 3. Exact duplicates
    df, removed = remove_exact_duplicates(df)
    report["exact_duplicates_removed"] = removed

    # 4. Duplicate image IDs
    df, removed = remove_duplicate_image_ids(df)
    report["duplicate_image_ids_removed"] = removed

    # 5. Invalid labels
    df, removed = filter_valid_classes(df)
    report["invalid_class_rows_removed"] = removed

    # 6. Match metadata with image files
    image_map = collect_image_paths(raw_data_dir)

    report["image_files_found"] = len(image_map)

    df = attach_image_paths(df, image_map)

    # 7. Missing image files
    df, removed = remove_missing_image_files(df)
    report["missing_image_rows_removed"] = removed

    # 8. Corrupted images
    if verify_images:
        df, corrupted = validate_images(df)
    else:
        corrupted = []

    report["corrupted_images_removed"] = len(corrupted)
    report["corrupted_images"] = corrupted

    # Reset index
    df = df.reset_index(drop=True)

    report["final_rows"] = len(df)
    report["rows_removed_total"] = (
        report["initial_rows"] -
        report["final_rows"]
    )

    return df, report
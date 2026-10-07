import pandas as pd
import pytest

from src.data.splitting import (
    check_image_leakage,
    check_lesion_leakage,
    stratified_lesion_split,
    validate_split_integrity,
)

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_synthetic_metadata(lesions_per_class: int = 20) -> pd.DataFrame:
    rows = []
    for class_name in CLASSES:
        for lesion_index in range(lesions_per_class):
            lesion_id = f"{class_name}_lesion_{lesion_index:03d}"
            for image_index in range(2):
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": f"{lesion_id}_img_{image_index}",
                        "dx": class_name,
                    }
                )
    return pd.DataFrame(rows)


def test_stratified_lesion_split_has_no_lesion_or_image_leakage():
    df = make_synthetic_metadata()
    train_df, val_df, test_df = stratified_lesion_split(df, 0.70, 0.15, 0.15, 42)

    lesion_leakage = check_lesion_leakage(train_df, val_df, test_df)
    image_leakage = check_image_leakage(train_df, val_df, test_df)

    assert all(len(values) == 0 for values in lesion_leakage.values())
    assert all(len(values) == 0 for values in image_leakage.values())


def test_stratified_lesion_split_preserves_all_images():
    df = make_synthetic_metadata()
    train_df, val_df, test_df = stratified_lesion_split(df, random_state=42)
    report = validate_split_integrity(df, train_df, val_df, test_df)

    assert report["all_images_preserved"]
    assert report["same_image_set"]
    assert report["no_lesion_leakage"]
    assert report["no_image_leakage"]


def test_stratified_lesion_split_is_reproducible():
    df = make_synthetic_metadata()
    split_a = stratified_lesion_split(df, random_state=42)
    split_b = stratified_lesion_split(df, random_state=42)

    for dataframe_a, dataframe_b in zip(split_a, split_b):
        assert dataframe_a["image_id"].tolist() == dataframe_b["image_id"].tolist()
        assert dataframe_a["lesion_id"].tolist() == dataframe_b["lesion_id"].tolist()


def test_stratified_lesion_split_rejects_invalid_split_sum():
    df = make_synthetic_metadata()
    with pytest.raises(ValueError, match="must equal 1.0"):
        stratified_lesion_split(df, train_size=0.80, val_size=0.15, test_size=0.15, random_state=42)

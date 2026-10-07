from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_ham10000
from src.data.splitting import (
    get_class_distribution,
    get_split_summary,
    save_splits,
    stratified_lesion_split,
    validate_split_integrity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean HAM10000 metadata and create a fixed lesion-aware split.")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing train/val/test split files.")
    return parser.parse_args()


def make_json_serializable(value):
    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = [output_dir / "train.csv", output_dir / "val.csv", output_dir / "test.csv"]

    if all(path.exists() for path in split_paths) and not args.force:
        print("Existing fixed split detected:")
        for path in split_paths:
            print(f"  - {path}")
        print("No split files were overwritten. Use --force only if you intentionally want to regenerate the split.")
        return

    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata rows: {len(metadata):,}")

    cleaned_df, cleaning_report = clean_ham10000(
        df=metadata,
        raw_data_dir=raw_dir,
        verify_images=not args.skip_image_verification,
    )

    cleaned_df.to_csv(output_dir / "cleaned_metadata.csv", index=False)
    with (output_dir / "cleaning_report.json").open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(cleaning_report), file, indent=2, ensure_ascii=False, default=str)

    train_df, val_df, test_df = stratified_lesion_split(
        cleaned_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    integrity = validate_split_integrity(cleaned_df, train_df, val_df, test_df)
    if not integrity["no_lesion_leakage"]:
        raise RuntimeError("Lesion leakage detected. Split files were not saved.")
    if not integrity["no_image_leakage"]:
        raise RuntimeError("Image leakage detected. Split files were not saved.")
    if not integrity["all_images_preserved"] or not integrity["same_image_set"]:
        raise RuntimeError("Split integrity check failed. Split files were not saved.")

    save_splits(train_df, val_df, test_df, output_dir)
    split_summary = get_split_summary(train_df, val_df, test_df)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        get_class_distribution(split_df).to_csv(output_dir / f"{split_name}_class_distribution.csv")

    with (output_dir / "split_integrity.json").open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(integrity), file, indent=2, ensure_ascii=False)

    print("\nData preparation completed.")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(split_summary.to_string(index=False))
    print("Lesion leakage: none")
    print("Image leakage: none")
    print(f"Processed files saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""
download_data.py - fetch a REAL cats & dogs dataset.

Source: Microsoft "Dogs vs Cats" (Kaggle competition data) mirrored on
Hugging Face as `microsoft/cats_vs_dogs` (~25,000 real photos).

This script streams the dataset and saves `IMAGES_PER_CLASS` images per class
into:
    data/cats/   (label 0)
    data/dogs/   (label 1)

Streaming mode downloads only the rows we keep, so it is disk friendly.

Usage:
    python download_data.py [--num-images 1200] [--dataset microsoft/cats_vs_dogs]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config


def download_sample_images(dataset_dir=config.DATA_DIR,
                           num_per_class=config.IMAGES_PER_CLASS,
                           hf_dataset=config.HF_DATASET) -> int:
    """Stream `num_per_class` images per class from the Hugging Face dataset."""
    cats_dir = os.path.join(dataset_dir, "cats")
    dogs_dir = os.path.join(dataset_dir, "dogs")
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split="train", streaming=True)

    counts = {"cat": 0, "dog": 0}
    total = 0

    print(f"[INFO] Streaming '{hf_dataset}' - {num_per_class} images per class...")
    for row in ds:
        seed = row.get("image")
        if seed is None:
            continue
        label = row.get("labels", row.get("label"))
        # label values are 0 (cat) / 1 (dog); names may be available too
        if isinstance(label, str):
            name = label.lower()
        else:
            name = "dog" if int(label) == 1 else "cat"
        if counts[name] >= num_per_class:
            continue

        out = os.path.join(cats_dir if name == "cat" else dogs_dir,
                           f"{name}_{counts[name]+1}.jpg")
        img = seed.convert("RGB") if hasattr(seed, "convert") else seed
        img.save(out, "JPEG")

        counts[name] += 1
        total += 1
        if total % 200 == 0:
            print(f"[..] cats={counts['cat']} dogs={counts['dog']}")

        if all(v >= num_per_class for v in counts.values()):
            break

    print(f"[DONE] cats={counts['cat']} dogs={counts['dog']} saved to {dataset_dir}")
    # The `datasets` streaming layer can crash during interpreter shutdown
    # (harmless - all files are already written). Bypass the noisy teardown.
    os._exit(0)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download real cats/dogs dataset")
    parser.add_argument("--num-images", type=int, default=config.IMAGES_PER_CLASS,
                        help="images to fetch per class")
    parser.add_argument("--dataset", type=str, default=config.HF_DATASET,
                        help="Hugging Face dataset id")
    args = parser.parse_args(argv)
    download_sample_images(num_per_class=args.num_images, hf_dataset=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
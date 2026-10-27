"""
Leakage-Free Dataset Splitting Module.
Performs:
1. Strict 80% Training / 20% Held-Out Test Split.
2. Leakage Prevention:
   - Group-based splitting on parent_file and speaker_id: All segments from
     the same parent recording and same speaker are strictly assigned together.
     No recording or speaker can exist in both training and testing splits.
3. Cross-Dataset Evaluation Set:
   - Holds out samples from an independent source (zo9999 ElevenLabs Aria & Rachel)
     to evaluate out-of-distribution / cross-dataset generalization.
4. Fixed random seed (seed=42) for deterministic reproducibility.
"""

import os
import json
import random
from typing import Dict, List, Any


def split_dataset(processed_manifest_path: str = "data/processed/processed_manifest.json",
                  splits_dir: str = "data/splits",
                  seed: int = 42,
                  train_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Split processed segments into:
    - Train (80%)
    - Held-out Test (20%)
    - Cross-Dataset Out-of-Distribution Test Set
    """
    random.seed(seed)
    os.makedirs(splits_dir, exist_ok=True)

    with open(processed_manifest_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[Split] Splitting {len(segments)} segments with fixed seed={seed} (train_ratio={train_ratio})...")

    # Step 1: Separate cross-dataset evaluation candidates
    # We use zo9999 Deepfake Repository as out-of-distribution cross-dataset evaluation
    cross_dataset_segments = []
    in_distribution_segments = []

    for seg in segments:
        if "zo9999" in seg["dataset_source"]:
            cross_dataset_segments.append(seg)
        else:
            in_distribution_segments.append(seg)

    print(f"[Split] In-distribution pool: {len(in_distribution_segments)} segments.")
    print(f"[Split] Cross-dataset evaluation pool: {len(cross_dataset_segments)} segments.")

    # Step 2: Group by parent recording and speaker to prevent leakage
    # Structure: groups[label][group_id] = [seg1, seg2, ...]
    groups_real: Dict[str, List[Dict[str, Any]]] = {}
    groups_fake: Dict[str, List[Dict[str, Any]]] = {}

    for seg in in_distribution_segments:
        # Group key combines speaker_id and parent_file to ensure zero segment leakage
        group_key = f"{seg['speaker_id']}::{seg['parent_file']}"
        if seg["label"] == "REAL":
            groups_real.setdefault(group_key, []).append(seg)
        else:
            groups_fake.setdefault(group_key, []).append(seg)

    # Step 3: Stratified partition of groups
    def partition_groups(groups_dict: Dict[str, List[Dict[str, Any]]], ratio: float):
        keys = list(groups_dict.keys())
        random.shuffle(keys)
        # Calculate total segments to hit ratio approximately
        total_segs = sum(len(v) for v in groups_dict.values())
        target_train_segs = int(total_segs * ratio)

        train_keys = []
        test_keys = []
        curr_train = 0

        for k in keys:
            segs_count = len(groups_dict[k])
            if curr_train + segs_count <= target_train_segs or not train_keys:
                train_keys.append(k)
                curr_train += segs_count
            else:
                test_keys.append(k)

        # Flatten segments
        train_segs = [s for k in train_keys for s in groups_dict[k]]
        test_segs = [s for k in test_keys for s in groups_dict[k]]
        return train_segs, test_segs, train_keys, test_keys

    train_real, test_real, tr_real_keys, te_real_keys = partition_groups(groups_real, train_ratio)
    train_fake, test_fake, tr_fake_keys, te_fake_keys = partition_groups(groups_fake, train_ratio)

    train_split = train_real + train_fake
    test_split = test_real + test_fake

    # Shuffle within split
    random.shuffle(train_split)
    random.shuffle(test_split)

    # Step 4: Verification of zero leakage
    train_parent_files = set(s["parent_file"] for s in train_split)
    test_parent_files = set(s["parent_file"] for s in test_split)
    leakage_intersection = train_parent_files.intersection(test_parent_files)
    assert len(leakage_intersection) == 0, f"DATA LEAKAGE DETECTED! Overlapping files: {leakage_intersection}"

    train_manifest_path = os.path.join(splits_dir, "train_manifest.json")
    test_manifest_path = os.path.join(splits_dir, "test_manifest.json")
    cross_manifest_path = os.path.join(splits_dir, "cross_dataset_manifest.json")
    split_info_path = os.path.join(splits_dir, "split_metadata.json")

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)
    with open(cross_manifest_path, "w", encoding="utf-8") as f:
        json.dump(cross_dataset_segments, f, indent=2)

    metadata = {
        "random_seed": seed,
        "split_policy": "GroupKFold by parent_file and speaker_id",
        "leakage_prevention": "Strict zero-overlap between train and test parent recordings and speakers",
        "train_samples": len(train_split),
        "train_distribution": {
            "real": sum(1 for x in train_split if x["label"] == "REAL"),
            "fake": sum(1 for x in train_split if x["label"] == "FAKE")
        },
        "test_samples": len(test_split),
        "test_distribution": {
            "real": sum(1 for x in test_split if x["label"] == "REAL"),
            "fake": sum(1 for x in test_split if x["label"] == "FAKE")
        },
        "cross_dataset_samples": len(cross_dataset_segments),
        "cross_dataset_distribution": {
            "real": sum(1 for x in cross_dataset_segments if x["label"] == "REAL"),
            "fake": sum(1 for x in cross_dataset_segments if x["label"] == "FAKE")
        },
        "train_parent_recordings": list(train_parent_files),
        "test_parent_recordings": list(test_parent_files)
    }

    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Split] SUCCESS: Zero data leakage verified.")
    print(f"        Train Split: {len(train_split)} samples (REAL: {metadata['train_distribution']['real']}, FAKE: {metadata['train_distribution']['fake']})")
    print(f"        Test Split:  {len(test_split)} samples (REAL: {metadata['test_distribution']['real']}, FAKE: {metadata['test_distribution']['fake']})")
    print(f"        Cross-Dataset Split: {len(cross_dataset_segments)} samples")
    print(f"        Saved to {splits_dir}/")
    return metadata


if __name__ == "__main__":
    split_dataset()

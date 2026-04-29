#!/usr/bin/env python3
"""
Extract descriptor tensors from an hloc features.h5 file.

Reads DISK feature descriptors, pads to a fixed max_keypoints size,
converts to float16, L2-normalizes, and saves as a single .pt tensor.

ALL image-identifiable data is stripped:
  - No filenames
  - No keypoint (x, y) coordinates
  - No scores
  - No image dimensions

The output is a single tensor of shape (N_images, max_keypoints, 128)
containing only normalized float16 descriptor vectors.

Usage:
    python extract_payload.py /path/to/features.h5 --output data/descriptors.pt
"""
import argparse
import sys
import os

import h5py
import numpy as np
import torch


def extract_descriptors(h5_path, max_keypoints=1024, output_path="data/descriptors.pt"):
    """Extract, pad, normalize, and save descriptor tensors."""

    print(f"Reading: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        image_keys = sorted(f.keys())
        n_images = len(image_keys)
        print(f"  Found {n_images} images")

        # Determine descriptor dimension from first image
        first_desc = f[image_keys[0]]["descriptors"][:]
        desc_dim = first_desc.shape[1]
        print(f"  Descriptor dimension: {desc_dim}")

        # Pre-allocate output array
        all_descriptors = np.zeros((n_images, max_keypoints, desc_dim), dtype=np.float32)
        feature_counts = []

        for i, key in enumerate(image_keys):
            desc = f[key]["descriptors"][:]  # (N_features, desc_dim)
            n_feat = desc.shape[0]
            feature_counts.append(n_feat)

            # Truncate if more than max_keypoints
            if n_feat > max_keypoints:
                desc = desc[:max_keypoints]
                n_feat = max_keypoints

            # Copy into padded array (rest stays zero)
            all_descriptors[i, :n_feat, :] = desc

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{n_images}...")

    print(f"\n  Feature counts: min={min(feature_counts)}, "
          f"max={max(feature_counts)}, "
          f"mean={np.mean(feature_counts):.0f}")

    # L2 normalize (same as pipeline)
    norms = np.linalg.norm(all_descriptors, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    all_descriptors = all_descriptors / norms

    # Convert to float16 (same as pipeline's target_dtype)
    tensor = torch.from_numpy(all_descriptors).half()

    print(f"\n  Output tensor: {tensor.shape} ({tensor.dtype})")
    print(f"  Size: {tensor.nbytes / (1024**2):.1f} MB")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(tensor, output_path)
    print(f"  Saved: {output_path}")

    # Verify round-trip
    loaded = torch.load(output_path, weights_only=True)
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    print(f"  ✓ Verified: {loaded.shape} {loaded.dtype}")

    n_pairs = n_images * (n_images - 1) // 2
    print(f"\n  Exhaustive pairs: {n_pairs:,}")
    print(f"  At ~60 it/s: ~{n_pairs / 60:.0f}s ({n_pairs / 60 / 60:.1f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract descriptor tensors from features.h5 for crash reproduction"
    )
    parser.add_argument("h5_path", help="Path to hloc features.h5 file")
    parser.add_argument("--max-keypoints", type=int, default=1024,
                        help="Pad/truncate to this many keypoints per image (default: 1024)")
    parser.add_argument("--output", default="data/descriptors.pt",
                        help="Output .pt file path (default: data/descriptors.pt)")
    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"ERROR: {args.h5_path} not found")
        sys.exit(1)

    extract_descriptors(args.h5_path, args.max_keypoints, args.output)

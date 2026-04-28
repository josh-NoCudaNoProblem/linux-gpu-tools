#!/usr/bin/env python3
"""
rdna4_repro.py — Minimal reproducer for gfx1201 UTCL2 TLB coherency fault

Reproduces GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031 on AMD Radeon RX 9070 XT
by running LightGlue feature matching in a tight loop. No GUI, no pipeline,
no file I/O — just the bare inference loop that triggers the fault.

The fault occurs in `label_LoopBeginL` — a rocBLAS/HIP BLAS kernel used for
batched matrix multiplication (attention) inside LightGlue.

Usage:
    cd ~/Desktop/3DGS/"Pipeline V4"
    source venv-rocm/bin/activate

    # With real DISK features (recommended — proven to trigger the fault):
    python amd_debug/rdna4_repro.py --features /path/to/features.h5

    # With synthetic data (may not trigger the fault):
    python amd_debug/rdna4_repro.py --max-kp 1024 --pairs 600000

After crash, check:
    sudo dmesg | grep GCVM_L2_PROTECTION_FAULT_STATUS

Expected fault:
    GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
    Faulty UTCL2 client ID: TCP (0x8)
    PERMISSION_FAULTS: 0x3

Environment:
    GPU:   AMD Radeon RX 9070 XT (gfx1201 / RDNA4)
    ROCm:  7.2.1
    OS:    Fedora 43+ (kernel 6.19+)

Author: Joshua (forensic reproduction for AMD bug report)
Date:   2026-04-27
"""

import sys
import os
import time
import argparse
import signal

def main():
    parser = argparse.ArgumentParser(description="RDNA4 UTCL2 TLB fault reproducer")
    parser.add_argument("--features", type=str, default=None, help="Path to features.h5 (real DISK features)")
    parser.add_argument("--max-kp", type=int, default=1024, help="Max keypoints per image (default: 1024)")
    parser.add_argument("--pairs", type=int, default=0, help="Number of pairs (0 = exhaustive)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (crashes in both modes)")
    parser.add_argument("--desc-dim", type=int, default=128, help="Descriptor dimension (default: 128 for DISK)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[1920, 1080], help="Image size WxH (default: 1920 1080)")
    args = parser.parse_args()

    # ---- Import torch + lightglue ----
    print("Loading PyTorch...")
    import torch
    import numpy as np

    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP not available. Need ROCm-enabled PyTorch.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print()

    # ---- Load LightGlue ----
    print("Loading LightGlue...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    lightglue_dir = os.path.join(pipeline_dir, "LightGlue")
    for p in [pipeline_dir, lightglue_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from lightglue import LightGlue

    matcher = LightGlue(features="disk").eval().to(device).half()
    print("LightGlue loaded (FP16)")

    if args.compile:
        print("Compiling with torch.compile...")
        matcher = torch.compile(matcher, mode='default')
        print("torch.compile done")

    dtype = torch.float16

    # ---- Load features ----
    if args.features:
        import h5py
        print(f"Loading real features from {args.features}...")
        f_in = h5py.File(args.features, 'r')
        image_names = list(f_in.keys())

        all_features = {}
        for name in image_names:
            all_features[name] = {
                'keypoints': f_in[name]['keypoints'][:],
                'scores': f_in[name]['scores'][:],
                'descriptors': f_in[name]['descriptors'][:],
            }
            if 'image_size' in f_in[name]:
                all_features[name]['image_size'] = f_in[name]['image_size'][:]
        f_in.close()

        num_images = len(image_names)
        sample = all_features[image_names[0]]
        desc_dim = sample['descriptors'].shape[1]
        max_kp = args.max_kp

        # Get image size
        if 'image_size' in sample:
            img_w, img_h = int(sample['image_size'][0]), int(sample['image_size'][1])
        else:
            img_w, img_h = args.img_size

        total_bytes = sum(
            v['keypoints'].nbytes + v['scores'].nbytes + v['descriptors'].nbytes
            for v in all_features.values()
        )
        print(f"Loaded {num_images} images, {desc_dim}D descriptors, {total_bytes / (1024**2):.0f} MB")

        # Build exhaustive pairs
        pairs = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                pairs.append((image_names[i], image_names[j]))
        total = len(pairs) if args.pairs == 0 else min(args.pairs, len(pairs))
        use_real_data = True
    else:
        # Synthetic mode
        img_w, img_h = args.img_size
        desc_dim = args.desc_dim
        max_kp = args.max_kp
        num_images = 200

        print("Generating random feature pool...")
        cpu_features = []
        for i in range(num_images):
            kp = torch.rand(1, max_kp, 2, dtype=dtype) * torch.tensor([[[img_w, img_h]]], dtype=dtype)
            desc = torch.randn(1, max_kp, desc_dim, dtype=dtype)
            desc = desc / (desc.norm(dim=-1, keepdim=True) + 1e-6)
            cpu_features.append((kp, desc))
        print(f"Generated {num_images} synthetic images ({max_kp} kp each)")

        total = args.pairs if args.pairs > 0 else 600000
        use_real_data = False

    # ---- Pre-allocate GPU buffers (matching pipeline's exact pattern) ----
    real_size = torch.tensor([[img_w, img_h]], dtype=dtype, device=device)

    gpu_buf0 = {
        'keypoints':   torch.zeros((1, max_kp, 2),        dtype=dtype, device=device),
        'descriptors': torch.zeros((1, max_kp, desc_dim),  dtype=dtype, device=device),
        'image_size':  real_size,
    }
    gpu_buf1 = {
        'keypoints':   torch.zeros((1, max_kp, 2),        dtype=dtype, device=device),
        'descriptors': torch.zeros((1, max_kp, desc_dim),  dtype=dtype, device=device),
        'image_size':  real_size,
    }

    buf_vram = sum(v.nbytes for v in gpu_buf0.values()) + sum(v.nbytes for v in gpu_buf1.values())
    print(f"Pre-allocated GPU buffers: {buf_vram / (1024**2):.1f} MB")
    print()

    # ---- Config summary ----
    print("=" * 50)
    print(f"RDNA4 UTCL2 TLB Fault Reproducer")
    print(f"  Data:          {'REAL (' + str(num_images) + ' images)' if use_real_data else 'SYNTHETIC'}")
    print(f"  Max keypoints: {max_kp}")
    print(f"  Image size:    {img_w}x{img_h}")
    print(f"  Desc dim:      {desc_dim}")
    print(f"  Pairs to run:  {total}")
    print(f"  Mode:          {'torch.compile' if args.compile else 'eager'}")
    print(f"  Expected: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031")
    print("=" * 50)
    print()

    # ---- Matching loop (exact pattern from v4_backend.py) ----
    t0 = time.perf_counter()
    last_print = t0

    with torch.inference_mode():
        for i in range(total):
            if use_real_data:
                name0, name1 = pairs[i]
                feat0 = all_features[name0]
                feat1 = all_features[name1]
                n0 = min(feat0['keypoints'].shape[0], max_kp)
                n1 = min(feat1['keypoints'].shape[0], max_kp)

                kp0 = torch.from_numpy(feat0['keypoints'][:n0]).unsqueeze(0).to(dtype=dtype)
                desc0 = torch.from_numpy(feat0['descriptors'][:n0]).unsqueeze(0).to(dtype=dtype)
                kp1 = torch.from_numpy(feat1['keypoints'][:n1]).unsqueeze(0).to(dtype=dtype)
                desc1 = torch.from_numpy(feat1['descriptors'][:n1]).unsqueeze(0).to(dtype=dtype)
            else:
                idx0 = i % num_images
                idx1 = (i + 1 + (i // num_images)) % num_images
                if idx0 == idx1:
                    idx1 = (idx1 + 1) % num_images
                kp0, desc0 = cpu_features[idx0]
                kp1, desc1 = cpu_features[idx1]
                n0 = max_kp
                n1 = max_kp

            # Copy into pre-allocated GPU buffers (same as pipeline)
            gpu_buf0['keypoints'].zero_()
            gpu_buf0['keypoints'][:, :n0, :].copy_(kp0)
            gpu_buf0['descriptors'].zero_()
            gpu_buf0['descriptors'][:, :n0, :].copy_(desc0)

            gpu_buf1['keypoints'].zero_()
            gpu_buf1['keypoints'][:, :n1, :].copy_(kp1)
            gpu_buf1['descriptors'].zero_()
            gpu_buf1['descriptors'][:, :n1, :].copy_(desc1)

            # Run LightGlue inference
            input0 = {k: v for k, v in gpu_buf0.items()}
            input1 = {k: v for k, v in gpu_buf1.items()}

            result = matcher({'image0': input0, 'image1': input1})

            # Pull results back to CPU (triggers hipMemcpy — where the fault surfaces)
            matches = result['matches'][0].cpu()
            scores = result['scores'][0].cpu()
            del result

            # Progress
            now = time.perf_counter()
            if now - last_print >= 5.0:
                elapsed = now - t0
                its = (i + 1) / elapsed
                eta = (total - i - 1) / its if its > 0 else 0
                print(f"[{elapsed:8.1f}s] {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                      f"{its:.1f} it/s | ETA: {eta:.0f}s")
                last_print = now

    # If we get here, no fault occurred
    elapsed = time.perf_counter() - t0
    print(f"\nCompleted {total} pairs in {elapsed:.1f}s without fault.")
    print("Check: sudo dmesg | grep GCVM_L2_PROTECTION_FAULT_STATUS")


if __name__ == "__main__":
    main()

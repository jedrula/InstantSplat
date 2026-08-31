#!/usr/bin/env python3
"""
LangSplat Phase 2: Train a 512→3→512 autoencoder on extracted CLIP features.

Runs with instantsplat env:
  python train_autoencoder.py --feature-dir ... --output-dir ...

After training:
  - Saves autoencoder checkpoint (encoder + decoder weights)
  - Pre-encodes all feature maps to (H//4, W//4, 3) .npy files for training use
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


class LangAutoencoder(nn.Module):
    def __init__(self, feat_dim=512, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def iter_pixel_batches(feature_files, batch_size=4096, shuffle_files=True):
    """
    Yields (batch_size, C) tensors by loading each file once and sampling pixels.
    Memory footprint: one feature map in RAM at a time (~130MB max).
    """
    files = list(feature_files)
    if shuffle_files:
        np.random.shuffle(files)
    for fpath in files:
        feat_map = np.load(fpath).astype(np.float32)  # (H, W, 512)
        H, W, C = feat_map.shape
        pixels = feat_map.reshape(-1, C)              # (H*W, 512)
        idx = np.random.permutation(len(pixels))
        for start in range(0, len(pixels), batch_size):
            chunk = pixels[idx[start:start + batch_size]]
            if len(chunk) < 2:
                continue
            yield torch.from_numpy(chunk)


def main():
    parser = argparse.ArgumentParser(description="LangSplat autoencoder training")
    parser.add_argument("--feature-dir", required=True, help="Dir with *_lang.npy feature maps")
    parser.add_argument("--output-dir", required=True, help="Dir to write checkpoint + compressed maps")
    parser.add_argument("--latent-dim", type=int, default=3, help="Bottleneck dimension (default 3)")
    parser.add_argument("--feat-dim", type=int, default=512, help="CLIP feature dimension (default 512)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size (pixel samples)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LangSplat AE] device={device}  latent_dim={args.latent_dim}")

    feature_files = sorted(Path(args.feature_dir).glob("*_lang.npy"))
    if not feature_files:
        print(f"ERROR: no *_lang.npy files in {args.feature_dir}")
        sys.exit(1)
    print(f"[LangSplat AE] Found {len(feature_files)} feature maps")

    files = [str(f) for f in feature_files]
    model = LangAutoencoder(feat_dim=args.feat_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"[LangSplat AE] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        # iter_pixel_batches loads one file at a time — safe memory footprint
        for batch in iter_pixel_batches(files, batch_size=args.batch_size):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}")

    ckpt_path = os.path.join(args.output_dir, "autoencoder.pth")
    torch.save({
        "model_state": model.state_dict(),
        "feat_dim": args.feat_dim,
        "latent_dim": args.latent_dim,
    }, ckpt_path)
    print(f"\n[LangSplat AE] Saved checkpoint to {ckpt_path}")

    # Pre-encode all feature maps to latent space (H//4, W//4, 3) float16
    print("[LangSplat AE] Pre-encoding feature maps to latent space...")
    model.eval()
    with torch.no_grad():
        for feat_file in tqdm(feature_files):
            feat_map = np.load(str(feat_file)).astype(np.float32)  # (H, W, 512)
            H, W, C = feat_map.shape
            feat_tensor = torch.from_numpy(feat_map.reshape(-1, C)).to(device)
            z = model.encode(feat_tensor)  # (H*W, 3)
            z_map = z.cpu().float().numpy().reshape(H, W, args.latent_dim)
            out_name = feat_file.stem.replace("_lang", "_lang3") + ".npy"
            out_path = os.path.join(args.output_dir, out_name)
            np.save(out_path, z_map.astype(np.float16))

    # Copy meta.json and update with output paths
    meta_src = os.path.join(args.feature_dir, "features_meta.json")
    if os.path.exists(meta_src):
        with open(meta_src) as f:
            meta = json.load(f)
        for entry in meta["images"]:
            feat_path = Path(entry["features"])
            lang3_name = feat_path.stem.replace("_lang", "_lang3") + ".npy"
            entry["features3"] = os.path.join(args.output_dir, lang3_name)
        meta["latent_dim"] = args.latent_dim
        meta["autoencoder"] = ckpt_path
        with open(os.path.join(args.output_dir, "lang_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print("[LangSplat AE] Done.")


if __name__ == "__main__":
    main()

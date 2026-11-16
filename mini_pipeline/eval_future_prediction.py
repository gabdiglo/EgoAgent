from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_CSV = "data/processed/wtour_pairs_1fps_gap5_test_clean.csv"
STAGE1_CKPT = "mini_pipeline/checkpoints/stage1_encoder_contrastive.pt"
STAGE2_CKPT = "mini_pipeline/checkpoints/stage2_future_predictor.pt"

OUT_DIR = Path("mini_pipeline/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = OUT_DIR / "future_prediction_results.json"
RESULTS_CSV = OUT_DIR / "future_prediction_per_sample.csv"

IMAGE_SIZE = 160
BATCH_SIZE = 16
MAX_TEST_SAMPLES = None  # metti 2000 se vuoi una evaluation più veloce


class PairDataset(Dataset):
    def __init__(self, csv_path, max_samples=None):
        df = pd.read_csv(csv_path)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(max_samples, random_state=123).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.current = df["current_frame"].tolist()
        self.future = df["future_frame"].tolist()
        self.video = df["video"].tolist()

        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.current)

    def _load_rgb(self, path):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.load()
            return img

    def __getitem__(self, idx):
        for attempt in range(20):
            j = (idx + attempt) % len(self.current)
            try:
                x = self._load_rgb(self.current[j])
                y = self._load_rgb(self.future[j])
                return {
                    "x": self.tf(x),
                    "y": self.tf(y),
                    "video": self.video[j],
                    "current_frame": self.current[j],
                    "future_frame": self.future[j],
                }
            except Exception as e:
                if attempt == 0:
                    print(
                        f"Skip bad pair: {self.current[j]} | {self.future[j]} | "
                        f"{type(e).__name__}: {e}"
                    )
                continue

        raise RuntimeError(f"Troppi frame corrotti vicino all'indice {idx}")


class VisualEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        h = self.backbone(x).flatten(1)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z


class FuturePredictor(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, z):
        z = self.net(z)
        z = F.normalize(z, dim=1)
        return z


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    print("Carico encoder:", STAGE1_CKPT)
    encoder = VisualEncoder().to(device)
    enc_ckpt = torch.load(STAGE1_CKPT, map_location=device)
    encoder.load_state_dict(enc_ckpt["model_state"])
    encoder.eval()

    print("Carico future predictor:", STAGE2_CKPT)
    predictor = FuturePredictor().to(device)
    pred_ckpt = torch.load(STAGE2_CKPT, map_location=device)
    predictor.load_state_dict(pred_ckpt["model_state"])
    predictor.eval()

    ds = PairDataset(TEST_CSV, max_samples=MAX_TEST_SAMPLES)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print("Test samples:", len(ds))

    total_n = 0

    baseline_cos_sum = 0.0
    model_cos_sum = 0.0

    baseline_mse_sum = 0.0
    model_mse_sum = 0.0

    per_sample_rows = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluation"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            z_now = encoder(x)
            z_future = encoder(y)
            z_pred = predictor(z_now)

            baseline_cos = F.cosine_similarity(z_now, z_future, dim=1)
            model_cos = F.cosine_similarity(z_pred, z_future, dim=1)

            baseline_mse = torch.mean((z_now - z_future) ** 2, dim=1)
            model_mse = torch.mean((z_pred - z_future) ** 2, dim=1)

            bsz = x.size(0)
            total_n += bsz

            baseline_cos_sum += baseline_cos.sum().item()
            model_cos_sum += model_cos.sum().item()

            baseline_mse_sum += baseline_mse.sum().item()
            model_mse_sum += model_mse.sum().item()

            for i in range(bsz):
                per_sample_rows.append({
                    "video": batch["video"][i],
                    "current_frame": batch["current_frame"][i],
                    "future_frame": batch["future_frame"][i],
                    "baseline_cosine": float(baseline_cos[i].cpu()),
                    "model_cosine": float(model_cos[i].cpu()),
                    "baseline_mse": float(baseline_mse[i].cpu()),
                    "model_mse": float(model_mse[i].cpu()),
                })

    baseline_cos_avg = baseline_cos_sum / total_n
    model_cos_avg = model_cos_sum / total_n

    baseline_mse_avg = baseline_mse_sum / total_n
    model_mse_avg = model_mse_sum / total_n

    results = {
        "num_test_samples": total_n,
        "baseline_cosine": baseline_cos_avg,
        "model_cosine": model_cos_avg,
        "baseline_mse": baseline_mse_avg,
        "model_mse": model_mse_avg,
        "cosine_improvement": model_cos_avg - baseline_cos_avg,
        "mse_improvement": baseline_mse_avg - model_mse_avg,
    }

    print("\nRISULTATI EVALUATION RIDOTTA")
    print("----------------------------")
    print(f"Test samples:        {results['num_test_samples']}")
    print(f"Baseline cosine:     {results['baseline_cosine']:.6f}")
    print(f"Model cosine:        {results['model_cosine']:.6f}")
    print(f"Cosine improvement:  {results['cosine_improvement']:.6f}")
    print()
    print(f"Baseline MSE:        {results['baseline_mse']:.6f}")
    print(f"Model MSE:           {results['model_mse']:.6f}")
    print(f"MSE improvement:     {results['mse_improvement']:.6f}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(per_sample_rows).to_csv(RESULTS_CSV, index=False)

    print("\nSalvati:")
    print(RESULTS_JSON)
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()

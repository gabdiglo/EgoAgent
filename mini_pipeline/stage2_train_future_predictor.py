from pathlib import Path

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

TRAIN_CSV = "data/processed/wtour_pairs_1fps_gap5_train_clean.csv"
VAL_CSV = "data/processed/wtour_pairs_1fps_gap5_val_clean.csv"
STAGE1_CKPT = "mini_pipeline/checkpoints/stage1_encoder_contrastive.pt"

OUT_DIR = Path("mini_pipeline/checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "stage2_future_predictor.pt"

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 3000


class PairDataset(Dataset):
    def __init__(self, csv_path, max_samples=None):
        df = pd.read_csv(csv_path)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42).reset_index(drop=True)

        self.current = df["current_frame"].tolist()
        self.future = df["future_frame"].tolist()

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
        # Alcuni frame possono essere corrotti/troncati.
        # Se una coppia fallisce, proviamo le successive.
        for attempt in range(20):
            j = (idx + attempt) % len(self.current)
            try:
                x = self._load_rgb(self.current[j])
                y = self._load_rgb(self.future[j])
                return self.tf(x), self.tf(y)
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


def run_epoch(encoder, predictor, loader, device, optimizer=None):
    train = optimizer is not None

    encoder.eval()
    predictor.train(train)

    total_loss = 0.0
    total_cos = 0.0
    total_n = 0
    total_batches = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            z_now = encoder(x)
            z_future = encoder(y)

        z_pred = predictor(z_now)

        cos = F.cosine_similarity(z_pred, z_future, dim=1)
        loss = 1.0 - cos.mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_cos += cos.sum().item()
        total_n += x.size(0)
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    avg_cos = total_cos / max(1, total_n)

    return avg_loss, avg_cos


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    print("Carico encoder Stage 1:", STAGE1_CKPT)

    encoder = VisualEncoder().to(device)
    ckpt = torch.load(STAGE1_CKPT, map_location=device)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    predictor = FuturePredictor().to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=LR)

    train_ds = PairDataset(TRAIN_CSV, max_samples=MAX_TRAIN_SAMPLES)
    val_ds = PairDataset(VAL_CSV, max_samples=MAX_VAL_SAMPLES)

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nStage 2 Epoch {epoch}/{EPOCHS}")

        train_loss, train_cos = run_epoch(
            encoder,
            predictor,
            train_loader,
            device,
            optimizer,
        )

        with torch.no_grad():
            val_loss, val_cos = run_epoch(
                encoder,
                predictor,
                val_loader,
                device,
                optimizer=None,
            )

        print(f"train_loss={train_loss:.4f} | train_cos={train_cos:.4f}")
        print(f"val_loss={val_loss:.4f} | val_cos={val_cos:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state": predictor.state_dict(),
                    "val_loss": val_loss,
                    "val_cos": val_cos,
                    "image_size": IMAGE_SIZE,
                    "emb_dim": 256,
                },
                OUT_PATH,
            )

            print("Salvato miglior checkpoint:", OUT_PATH)

    print("\nStage 2 completato.")
    print("Checkpoint:", OUT_PATH)


if __name__ == "__main__":
    main()

import math
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

OUT_DIR = Path("mini_pipeline/checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "stage1_encoder_contrastive.pt"

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 3000
TEMPERATURE = 0.1


class TemporalPairDataset(Dataset):
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
        # Alcuni frame estratti possono essere corrotti/troncati.
        # In quel caso saltiamo alla coppia successiva invece di bloccare il training.
        for attempt in range(20):
            j = (idx + attempt) % len(self.current)
            try:
                x = self._load_rgb(self.current[j])
                y = self._load_rgb(self.future[j])
                return self.tf(x), self.tf(y)
            except Exception as e:
                if attempt == 0:
                    print(f"Skip bad pair: {self.current[j]} | {self.future[j]} | {type(e).__name__}: {e}")
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


def contrastive_loss(z1, z2, temperature=0.1):
    """
    z1: embedding frame corrente
    z2: embedding frame futuro vicino
    Obiettivo: z1[i] vicino a z2[i], lontano da z2[j].
    """
    logits = z1 @ z2.T
    logits = logits / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_batches = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        z_x = model(x)
        z_y = model(y)
        loss = contrastive_loss(z_x, z_y, TEMPERATURE)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Device:", device)

    train_ds = TemporalPairDataset(TRAIN_CSV, max_samples=MAX_TRAIN_SAMPLES)
    val_ds = TemporalPairDataset(VAL_CSV, max_samples=MAX_VAL_SAMPLES)

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

    model = VisualEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = math.inf

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, optimizer=None)

        print(f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "emb_dim": 256,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                OUT_PATH,
            )
            print("Salvato miglior checkpoint:", OUT_PATH)

    print("\nStage 1 completato.")
    print("Checkpoint:", OUT_PATH)


if __name__ == "__main__":
    main()

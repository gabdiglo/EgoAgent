"""Strict evaluation of the reduced ResNet/MLP pipeline. No frame substitution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, csv, hashlib, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import resnet18

ImageFile.LOAD_TRUNCATED_IMAGES = False

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

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all test rows")
    parser.add_argument("--resume", action="store_true", help="Reuse saved embeddings for the same input and checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.project.resolve()
    out = args.output or root / "mini_pipeline/evaluation_strict"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    test_csv = root / "data/processed/wtour_pairs_1fps_gap5_test_clean.csv"
    with test_csv.open() as f:
        all_rows = list(csv.DictReader(f))
    selected = all_rows[:args.limit] if args.limit else all_rows
    unique, excluded, seen = [], [], set()
    for i, row in enumerate(selected):
        row = dict(row, source_row=i)
        key = (row["current_frame"], row["future_frame"])
        if key in seen:
            excluded.append(dict(row, reason="duplicate_pair"))
        else:
            seen.add(key)
            unique.append(row)
    enc_path = root / "mini_pipeline/checkpoints/stage1_encoder_contrastive.pt"
    pred_path = root / "mini_pipeline/checkpoints/stage2_future_predictor.pt"
    enc_ckpt = torch.load(enc_path, map_location="cpu", weights_only=True)
    pred_ckpt = torch.load(pred_path, map_location="cpu", weights_only=True)
    size, dim = enc_ckpt["image_size"], enc_ckpt["emb_dim"]
    if (size, dim) != (pred_ckpt["image_size"], pred_ckpt["emb_dim"]):
        raise ValueError("Checkpoint metadata mismatch")
    encoder, predictor = VisualEncoder(dim).to(device), FuturePredictor(dim).to(device)
    encoder.load_state_dict(enc_ckpt["model_state"])
    predictor.load_state_dict(pred_ckpt["model_state"])
    encoder.eval(); predictor.eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    paths = sorted({r[k] for r in unique for k in ["current_frame", "future_frame"]})
    def decode(name):
        try:
            with Image.open(root / name) as im:
                im.load()  # strict decode before resizing
                x = tf(im.convert("RGB"))
            return name, x, None
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            return name, None, f"{type(e).__name__}: {e}"
    cache, errors = {}, {}
    progress_path = out / "embedding_progress.pt"
    signature = {"test":digest(test_csv), "encoder":digest(enc_path), "size":size, "dim":dim, "limit":args.limit, "strict_decode":True}
    accumulated_seconds = 0.0
    if args.resume and progress_path.exists():
        state = torch.load(progress_path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Saved progress does not match current inputs")
        cache, errors = state["cache"], state["errors"]
        accumulated_seconds = state["elapsed_seconds"]
    pending = [p for p in paths if p not in cache and p not in errors]
    start = time.perf_counter()
    print(f"Device: {device}; selected pairs: {len(selected)}; unique frames: {len(paths)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for offset in range(0, len(pending), 128):
            good = []
            for name, x, error in pool.map(decode, pending[offset:offset+128]):
                if error:
                    errors[name] = error
                else:
                    good.append((name, x))
            for j in range(0, len(good), args.batch_size):
                batch = good[j:j+args.batch_size]
                z = encoder(torch.stack([x for _, x in batch]).to(device)).cpu()
                if not torch.isfinite(z).all():
                    raise ValueError("Non-finite encoder output")
                cache.update({name: v.clone() for (name, _), v in zip(batch, z)})
            if offset % 512 == 0 or offset + 128 >= len(pending):
                elapsed = accumulated_seconds + time.perf_counter()-start
                temporary = progress_path.with_suffix(".tmp")
                torch.save({"signature":signature, "cache":cache, "errors":errors, "elapsed_seconds":elapsed}, temporary)
                temporary.replace(progress_path)
                print(f"Frames checked: {len(cache)+len(errors)}/{len(paths)}; invalid: {len(errors)}; elapsed: {elapsed:.1f}s", flush=True)
        valid = []
        for row in unique:
            bad = [f"{row[k]}: {errors[row[k]]}" for k in ["current_frame", "future_frame"] if row[k] in errors]
            if bad:
                excluded.append(dict(row, reason="; ".join(bad)))
            else:
                valid.append(row)
        metrics = []
        for j in range(0, len(valid), args.batch_size):
            batch = valid[j:j+args.batch_size]
            x = torch.stack([cache[r["current_frame"]] for r in batch]).to(device)
            y = torch.stack([cache[r["future_frame"]] for r in batch]).to(device)
            pred = predictor(x)
            values = torch.stack([F.cosine_similarity(x,y), F.cosine_similarity(pred,y), ((x-y)**2).mean(1), ((pred-y)**2).mean(1)], dim=1).cpu().tolist()
            for row, vals in zip(batch, values):
                metrics.append(dict(row, **dict(zip(["baseline_cosine","model_cosine","baseline_mse","model_mse"],vals))))
    columns = ["video", "current_frame", "future_frame", "gap_seconds", "source_row"]
    write_csv(out/"excluded_pairs.csv", excluded, columns+["reason"])
    write_csv(out/"invalid_frames.csv", [{"frame":k,"reason":v} for k,v in errors.items()], ["frame","reason"])
    write_csv(out/"per_sample.csv", metrics, columns+["baseline_cosine","model_cosine","baseline_mse","model_mse"])
    if not metrics:
        raise ValueError("No valid pairs; inspect excluded_pairs.csv")
    means = {k: float(np.mean([r[k] for r in metrics])) for k in ["baseline_cosine","model_cosine","baseline_mse","model_mse"]}
    result = dict(means, cosine_improvement=means["model_cosine"]-means["baseline_cosine"], mse_improvement=means["baseline_mse"]-means["model_mse"],
        num_csv_pairs=len(all_rows), num_selected_pairs=len(selected), num_evaluated_pairs=len(metrics), num_excluded_pairs=len(excluded), num_invalid_frames=len(errors), num_unique_frames=len(paths),
        full_test=not args.limit, device=device, image_size=size, emb_dim=dim, batch_size=args.batch_size, cpu_threads=4, decode_threads=4,
        evaluation_seconds=accumulated_seconds+time.perf_counter()-start, torch_version=torch.__version__, precision="float32", strict_image_decode=True, replacement_of_bad_pairs=False,
        test_csv_sha256=digest(test_csv), encoder_sha256=digest(enc_path), predictor_sha256=digest(pred_path),
        videos=sorted({r["video"] for r in metrics}), cosine_win_fraction=float(np.mean([r["model_cosine"]>r["baseline_cosine"] for r in metrics])))
    assert len(metrics)+len(excluded)==len(selected)
    assert len({(r["current_frame"],r["future_frame"]) for r in metrics})==len(metrics)
    (out/"results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()

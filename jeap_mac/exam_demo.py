"""Demo visiva mini-JEAP: nessuna generazione RGB, nessuna selezione del best-case."""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(__file__).parent / "cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from data import Clips
from model import JEAP, Config

def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=root/"data/prepared/val.jsonl")
    p.add_argument("--checkpoint", type=Path, default=root/"runs/joint/best.pt")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, default=root/"runs/exam_demo/visualizzazione")
    a = p.parse_args()
    torch.set_num_threads(2)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("training_mode") != "joint_real_RGB_3D_pose":
        raise ValueError("Serve un checkpoint joint su RGB e pose reali")
    m = JEAP(Config(**ckpt["configuration"]))
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    ds = Clips(a.manifest, m.c.image_size)
    if not 0 <= a.index < len(ds):
        p.error(f"Indice fuori intervallo [0,{len(ds)-1}]")
    row = ds.rows[a.index]
    item = ds[a.index]
    frames = item["frames"].unsqueeze(0)
    acts = item["actions"].unsqueeze(0)
    mask = item["mask"].unsqueeze(0)
    with torch.inference_mode():
        out = m(frames[:,:-1], acts, mask)
        f, _ = m.observer.encode(frames[:,1:].flatten(0,1))
        c, _ = m.observer.encode(frames[:,:-1].flatten(0,1))
        similarity = torch.nn.functional.cosine_similarity(out["state_logits"].flatten(0,1), f).numpy()
        baseline = torch.nn.functional.cosine_similarity(c,f).numpy()
    truth = acts[0].numpy(); prediction = out["actions"][0].numpy()
    visibility = mask[0].numpy()
    shared = visibility[1:] * visibility[:-1]
    n = float(shared.sum())
    if not n:
        raise ValueError("Nessun giunto valido nel confronto con la persistenza")
    mpjpe = float(100 * (np.linalg.norm(prediction[1:]-truth[1:], axis=-1)*shared).sum()/n)
    persistent = float(100 * (np.linalg.norm(truth[:-1]-truth[1:], axis=-1)*shared).sum()/n)
    a.output.mkdir(parents=True, exist_ok=True)
    steps = len(row["frames"])-1
    fig = plt.figure(figsize=(15,9),layout="constrained")
    grid=fig.add_gridspec(3,4,height_ratios=[1.1,1,1])
    for i,name in enumerate(row["frames"]):
        ax=fig.add_subplot(grid[0,i])
        with Image.open(a.manifest.resolve().parent/name) as im:
            im.load()
            ax.imshow(im.convert("RGB"))
        ax.set_title("Futuro reale" if i==steps else f"Osservato t={i}")
        ax.axis("off")
    ax=fig.add_subplot(grid[1,:2])
    xs=np.arange(steps)
    ax.bar(xs-.18,similarity,.36,label="Mini-JEAP")
    ax.bar(xs+.18,baseline,.36,label="Persistenza")
    ax.set_xticks(xs,[f"{i}→{i+1}" for i in xs])
    ax.set_ylabel("Coseno, logit DINO")
    ax.set_title("Stato futuro: confronto latente")
    ax.legend()
    ax=fig.add_subplot(grid[1,2:])
    ax.bar(["Mini-JEAP","Persistenza"],[mpjpe,persistent])
    ax.set_ylabel("MPJPE (cm) ↓")
    ax.set_title("Pose: confronto sul clip")
    ax=fig.add_subplot(grid[2,:2])
    j=steps-1
    valid=visibility[j]>0
    ax.scatter(truth[j,valid,0],truth[j,valid,1],label="Pose reale")
    ax.scatter(prediction[j,valid,0],prediction[j,valid,1],marker="x",label="Pose predetta")
    ax.set_xlabel("X camera (m)");ax.set_ylabel("Y camera (m)")
    ax.set_title("Giunti in coordinate camera (non pixel)")
    ax.legend();ax.axis("equal")
    ax=fig.add_subplot(grid[2,2:]);ax.axis("off")
    ax.text(.02,.95,
        f"Clip {a.index}/{len(ds)-1}; take {row.get('take_uid','?')}\n"
        f"Frame IDs: {row['frame_ids']}\n"
        f"Checkpoint: epoca {ckpt.get('epoch','?')}\n\n"
        "ATTENZIONE: il frame futuro e' reale, NON generato.\n"
        "EgoAgent prevede embedding; questa e' mini-JEAP.\n"
        "Valutazione sul manifest scelto (default: validazione).\n"
        "Coseno sui logit DINO, non Top-1/mAP del paper.",
        va="top",fontsize=10,linespacing=1.6)
    fig.suptitle("Demo per l'esame — mini-JEAP RGB + pose 3D",fontsize=17)
    png=a.output/f"clip_{a.index:03d}.png"
    fig.savefig(png,dpi=160,bbox_inches="tight");plt.close(fig)
    report=dict(model="mini-JEAP, not official EgoAgent",clip_index=a.index,
        sample_selection="fixed manifest order; no cherry-picking",
        future_image="real ground-truth, not generated",state_metric="DINO-logit cosine, not retrieval",
        state_cosine_model=[float(v) for v in similarity],
        state_cosine_persistence=[float(v) for v in baseline],
        MPJPE_cm=mpjpe,persistence_MPJPE_cm=persistent,
        checkpoint_epoch=ckpt.get("epoch"),take_uid=row.get("take_uid"),
        frame_ids=row["frame_ids"])
    (a.output/f"clip_{a.index:03d}.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"Creati: {png} e clip_{a.index:03d}.json")
if __name__=="__main__":
    main()

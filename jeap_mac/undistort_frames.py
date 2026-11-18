"""Undistort Ego-Exo4D Aria RGB frames using real VRS camera calibration.

Geometric fisheye-to-pinhole rectification (not AI enhancement).
Original frame indices and synchronized timestamps remain unchanged.
Requires authorized Project Aria data and projectaria-tools.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def list_frames(folder: Path) -> list[Path]:
    """Return frames sorted by numeric ID; fail if IDs are ambiguous."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory inesistente: {folder}")
    frames = [p for p in folder.iterdir() if p.is_file() and FRAME_RE.fullmatch(p.name)]
    if not frames:
        raise ValueError(f"Nessun frame_XXXXXXX.jpg/png trovato in {folder}")
    frames.sort(key=lambda p: int(FRAME_RE.fullmatch(p.name).group(1)))
    ids = [int(FRAME_RE.fullmatch(p.name).group(1)) for p in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("ID frame duplicati: la sincronizzazione sarebbe ambigua")
    return frames


def source_calibration(vrs_path: Path, frame_size: tuple[int, int], label: str):
    """Read VRS calibration and rescale it to existing frame-aligned MP4 dimensions."""
    try:
        from projectaria_tools.core import data_provider, calibration
    except ImportError as exc:
        raise RuntimeError(
            "Installa projectaria-tools: python -m pip install projectaria-tools"
        ) from exc
    if not vrs_path.is_file():
        raise FileNotFoundError(f"File VRS non trovato: {vrs_path}")
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    camera = provider.get_device_calibration().get_camera_calib(label)
    if camera is None:
        raise ValueError(f"Calibrazione '{label}' assente nel VRS")
    sensor_w, sensor_h = map(int, camera.get_image_size())
    width, height = frame_size
    sx, sy = width / sensor_w, height / sensor_h
    if abs(sx - sy) > 1e-4:
        raise ValueError(
            "Le immagini non preservano l'aspect ratio del sensore. "
            f"Frame {width}x{height}, calibrazione {sensor_w}x{sensor_h}; "
            "non applico una geometria potenzialmente errata."
        )
    src = camera.rescale([width, height], sx)
    return src, calibration


def rectify_one(rgb: np.ndarray, source, target, calibration, interpolation) -> np.ndarray:
    rectified = calibration.distort_by_calibration(rgb, target, source, interpolation)
    rectified = np.asarray(rectified)
    if rectified.shape != rgb.shape or rectified.dtype != np.uint8:
        raise ValueError("Project Aria ha prodotto un output con forma o dtype inattesi")
    return rectified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True,
                        help="Directory dei frame MP4 allineati (originali, distorti)")
    parser.add_argument("--vrs", type=Path, required=True,
                        help="VRS dello stesso take, contenente la calibrazione Aria")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory NUOVA per i frame rettificati")
    parser.add_argument("--camera-label", default="camera-rgb")
    parser.add_argument("--focal-scale", type=float, default=1.0,
                        help="Moltiplicatore della focale pinhole; default 1.0")
    parser.add_argument("--limit", type=int, default=0,
                        help="0=tutti i frame; >0 solo primi N (prova)")
    args = parser.parse_args()

    if args.focal_scale <= 0:
        parser.error("--focal-scale deve essere positivo")
    if args.limit < 0:
        parser.error("--limit non può essere negativo")

    source_dir = args.frames.resolve()
    target_dir = args.output.resolve()
    if source_dir == target_dir or source_dir in target_dir.parents or target_dir in source_dir.parents:
        parser.error("Input e output devono essere directory indipendenti; non sovrascrivo gli originali")

    frames = list_frames(source_dir)
    if args.limit:
        frames = frames[:args.limit]
    with Image.open(frames[0]) as im:
        im.load()
        width, height = im.size
    source, calibration = source_calibration(
        args.vrs.resolve(), (width, height), args.camera_label
    )
    focal = float(source.get_focal_lengths()[0]) * args.focal_scale
    target = calibration.get_linear_camera_calibration(
        width, height, focal, args.camera_label
    )
    from projectaria_tools.core.image import InterpolationMethod

    # Fail rather than silently reusing old results made with another calibration.
    if target_dir.exists() and any(target_dir.iterdir()):
        parser.error(f"Output non vuoto: {target_dir}. Scegli una directory nuova")
    target_dir.mkdir(parents=True, exist_ok=True)

    compared = False
    count = 0
    for path in frames:
        with Image.open(path) as im:
            im.load()
            if im.size != (width, height):
                raise ValueError(f"Frame di dimensione incoerente: {path.name}")
            rgb = np.array(im.convert("RGB"), dtype=np.uint8)
        dst = rectify_one(
            rgb, source, target, calibration, InterpolationMethod.BILINEAR
        )
        outfile = target_dir / path.name  # Important: preserve frame IDs.
        tmp = target_dir / (path.stem + ".tmp" + path.suffix)
        Image.fromarray(dst).save(tmp)
        tmp.replace(outfile)
        if not compared:
            preview = Image.new("RGB", (width * 2, height))
            preview.paste(Image.fromarray(rgb), (0, 0))
            preview.paste(Image.fromarray(dst), (width, 0))
            preview.save(target_dir / "before_after.jpg", quality=92)
            compared = True
        count += 1

    provenance = {
        "method": "Project Aria geometric fisheye-to-pinhole rectification",
        "source_frames": str(source_dir),
        "vrs": str(args.vrs.resolve()),
        "camera_label": args.camera_label,
        "frame_resolution": [width, height],
        "pinhole_focal_pixels": focal,
        "focal_scale": args.focal_scale,
        "frames_processed": count,
        "first_frame": frames[0].name,
        "last_frame": frames[-1].name,
        "original_frame_ids_preserved": True,
        "note": "Retains timestamps/frame IDs. This is not image denoising or learned enhancement.",
    }
    (target_dir / "rectification.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Rettificati {count} frame: {target_dir}")
    print(f"Anteprima: {target_dir / 'before_after.jpg'}")
    print("Per creare nuovi manifest, usa --frames con questa directory in prepare_clips.py")


if __name__ == "__main__":
    main()

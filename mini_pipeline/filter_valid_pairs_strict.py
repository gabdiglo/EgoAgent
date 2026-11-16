from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

CSV_FILES = [
    "data/processed/wtour_pairs_1fps_gap5_train.csv",
    "data/processed/wtour_pairs_1fps_gap5_val.csv",
    "data/processed/wtour_pairs_1fps_gap5_test.csv",
]

def can_load_rgb(path):
    path = Path(path)

    if not path.exists():
        return False

    if path.name.startswith("._"):
        return False

    if path.stat().st_size == 0:
        return False

    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.load()
        return True
    except Exception:
        return False

for csv_path in CSV_FILES:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    print(f"\nControllo severo: {csv_path}")

    valid = []
    bad = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        current_ok = can_load_rgb(row["current_frame"])
        future_ok = can_load_rgb(row["future_frame"])

        if current_ok and future_ok:
            valid.append(row)
        else:
            bad.append(row)

    clean_df = pd.DataFrame(valid)
    out_path = csv_path.with_name(csv_path.stem + "_clean.csv")
    clean_df.to_csv(out_path, index=False)

    print(f"Originali: {len(df)}")
    print(f"Valide:    {len(clean_df)}")
    print(f"Scartate:  {len(bad)}")
    print(f"Salvato:   {out_path}")

    if bad:
        print("\nEsempi problematici:")
        for r in bad[:10]:
            print("current:", r["current_frame"])
            print("future: ", r["future_frame"])
            print("---")

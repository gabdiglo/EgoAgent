from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

CSV_FILES = [
    "data/processed/wtour_pairs_1fps_gap5_train.csv",
    "data/processed/wtour_pairs_1fps_gap5_val.csv",
    "data/processed/wtour_pairs_1fps_gap5_test.csv",
]

def is_valid_image(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False

for csv_path in CSV_FILES:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    print(f"\nControllo: {csv_path}")
    valid_rows = []
    bad_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        ok_current = is_valid_image(row["current_frame"])
        ok_future = is_valid_image(row["future_frame"])

        if ok_current and ok_future:
            valid_rows.append(row)
        else:
            bad_rows.append(row)

    clean_df = pd.DataFrame(valid_rows)
    out_path = csv_path.with_name(csv_path.stem + "_clean.csv")
    clean_df.to_csv(out_path, index=False)

    print(f"Originali: {len(df)}")
    print(f"Valide:    {len(clean_df)}")
    print(f"Scartate:  {len(bad_rows)}")
    print(f"Salvato:   {out_path}")

    if bad_rows:
        print("Esempi righe problematiche:")
        for r in bad_rows[:5]:
            print(" current:", r["current_frame"])
            print(" future: ", r["future_frame"])

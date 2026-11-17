"""
Deler "Original Image" op i train / val / test (70 / 15 / 15).

Opdelingen er stratificeret på frugt OG klasse, så hver af de 16 grupper
(fx FreshBanana) fordeles ens: 140 til train, 30 til val og 30 til test.

Resultatet bliver denne mappestruktur:

    data/
      train/fresh/   train/rotten/
      val/fresh/     val/rotten/
      test/fresh/    test/rotten/

Filerne får frugtnavnet foran, fx "banana_IMG_0012.jpg", så man senere
kan se, hvilken frugt et billede er.

Kør én gang fra projektmappen:  python split_data.py
"""

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Indstillinger
SOURCE_DIR = Path("Original Image")   # mappen med de 16 undermapper
OUTPUT_DIR = Path("data")             # her lægges train/val/test
RANDOM_SEED = 42                      # fast seed = samme split hver gang

# Trin 0: Stop hvis data-mappen allerede findes
# (så vi ikke kommer til at blande to forskellige split sammen)
if OUTPUT_DIR.exists():
    print(f"Mappen '{OUTPUT_DIR}' findes allerede.")
    print("Slet den først, hvis du vil lave splittet igen.")
    raise SystemExit


# Trin 1: Lav en liste over alle billeder
rows = []

for folder in sorted(SOURCE_DIR.iterdir()):
    if not folder.is_dir():
        continue

    # Mappenavnet er fx "FreshBanana" eller "RottenBanana"
    if folder.name.startswith("Fresh"):
        label = "fresh"
        fruit = folder.name.replace("Fresh", "")
    else:
        label = "rotten"
        fruit = folder.name.replace("Rotten", "")

    for image_path in folder.iterdir():
        if image_path.is_file():
            rows.append({
                "path": image_path,
                "fruit": fruit.lower(),
                "label": label,
            })

df = pd.DataFrame(rows)
print(f"Fandt {len(df)} billeder")

# En kolonne der kombinerer frugt og klasse, fx "banana_rotten".
# Den bruges til at stratificere, så alle 16 grupper deles ens.
df["group"] = df["fruit"] + "_" + df["label"]


# Trin 2: Del op i train / val / test
# Først: 70 % train og 30 % "resten"
train_df, rest_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["group"],
    random_state=RANDOM_SEED,
)

# Derefter: "resten" deles i to lige store dele -> 15 % val og 15 % test
val_df, test_df = train_test_split(
    rest_df,
    test_size=0.50,
    stratify=rest_df["group"],
    random_state=RANDOM_SEED,
)

# Trin 3: Kopier billederne ud i de nye mapper
splits = {"train": train_df, "val": val_df, "test": test_df}

for split_name, split_df in splits.items():
    for _, row in split_df.iterrows():
        # fx data/train/rotten/
        target_folder = OUTPUT_DIR / split_name / row["label"]
        target_folder.mkdir(parents=True, exist_ok=True)

        # fx banana_IMG_0012.jpg
        new_name = f"{row['fruit']}_{row['path'].name}"
        shutil.copy(row["path"], target_folder / new_name)


# Trin 4: Tjek at tallene passer
# Forventet: train 1120/1120, val 240/240, test 240/240
print("\nAntal billeder i hver mappe:")
for split_name in splits:
    for label in ["fresh", "rotten"]:
        folder = OUTPUT_DIR / split_name / label
        count = len(list(folder.iterdir()))
        print(f"  {split_name:5} / {label:6}: {count}")

print("\nAntal billeder pr. frugt i testsættet (skal være 30 for alle):")
print(test_df.groupby(["fruit", "label"]).size().unstack().to_string())

# Gem hvilket billede der havnede hvor, så splittet kan dokumenteres
df["split"] = None
df.loc[train_df.index, "split"] = "train"
df.loc[val_df.index, "split"] = "val"
df.loc[test_df.index, "split"] = "test"
df.to_csv(OUTPUT_DIR / "split.csv", index=False)
print(f"\nOversigt gemt i {OUTPUT_DIR / 'split.csv'}")
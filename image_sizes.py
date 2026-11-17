"""
Tjekker billedstørrelser i "Original Image"-mappen fra
Fresh and Rotten Fruits-datasættet (Sultana et al., 2022).

Formålet er at se om friske og rådne billeder er forskellige i størrelse
eller format. Hvis de er, kan en model i princippet lære at skelne dem på
billedstørrelsen i stedet for på selve frugten, og det vil vi gerne vide
inden vi træner noget.

Kør:  python3 image_sizes.py "/sti/til/Original Image"
Kræver: pip install pillow pandas
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# Stien til datasættet gives som argument. Gives der ingen, bruges den mappe man står i.
data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")


# 1) Gå alle mapper og billeder igennem og saml info om hvert billede

# Hver frugt/klasse ligger i sin egen mappe. Vi læser frugt og klasse ud af mappenavnet og gemmer én række pr. billede i listen "rows".
rows = []
for folder in sorted(data_dir.iterdir()):
    if not folder.is_dir():
        continue

    # Mappenavne er fx "FreshApple" / "RottenApple"
    name = folder.name
    label = "fresh" if name.startswith("Fresh") else "rotten"
    fruit = name.removeprefix("Fresh").removeprefix("Rotten")

    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue
        try:
            # Image.open læser kun headeren, så det går hurtigt
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            # Ødelagte filer eller filer der ikke er billeder springes over,
            # men vi skriver dem ud så vi kan se hvor mange det drejer sig om
            print(f"Kunne ikke åbne: {img_path}")
            continue

        rows.append({
            "fruit": fruit,
            "label": label,
            "file_type": img_path.suffix.lower(),
            "width": width,
            "height": height,
            "aspect": round(width / height, 2),  # bredde/højde-forhold
        })


# 2) Lav det om til en tabel og gem den, så vi kan kigge på den senere
df = pd.DataFrame(rows)
df.to_csv("image_sizes.csv", index=False)


# 3) Udskriv et overblik over datasættet

# Hvor mange billeder er der, og hvilke filformater?
print(f"\nAntal billeder i alt: {len(df)}")
print("\nFiltyper:")
print(df["file_type"].value_counts().to_string())

# Hvor mange billeder er der af hver frugt i hver klasse? Bruges til at se om datasættet er skævt fordelt.
print("\nAntal billeder pr. frugt og klasse:")
print(df.groupby(["fruit", "label"]).size().unstack().to_string())

# Typiske størrelser pr. gruppe. Vi bruger median frem for gennemsnit, fordi nogle få meget store billeder ellers trækker tallet skævt.
print("\nMedian bredde x højde pr. frugt og klasse:")
summary = df.groupby(["fruit", "label"]).agg(
    n=("width", "size"),
    median_w=("width", "median"),
    median_h=("height", "median"),
    min_w=("width", "min"),
    max_w=("width", "max"),
    median_aspect=("aspect", "median"),
)
print(summary.to_string())


# 4) Det vigtigste tjek: er rådne og friske billeder lige store? Sammenlign friske og rådne billeder af samme frugt
print("\nForhold mellem rådden og frisk (median bredde):")
med = df.groupby(["fruit", "label"])["width"].median().unstack()
med["rotten/fresh"] = (med["rotten"] / med["fresh"]).round(2)
print(med.to_string())
print("\nTal langt fra 1.0 betyder, at friske og rådne billeder har forskellig "
      "størrelse, hvilket modellen potentielt kan udnytte som genvej.")
print("\nAlle målinger er gemt i image_sizes.csv")
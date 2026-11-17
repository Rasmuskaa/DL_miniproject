# Fresh vs. rotten fruit – ResNet-klassifikation

Miniprojekt i Deep Learning (DAKI, Aalborg Universitet). ResNet18, -34 og -50 (prætrænet på ImageNet og trænet fra bunden) finetunes til at klassificere frugt som **fresh** eller **rotten**. Grad-CAM bruges til at se, hvad modellen kigger på.

## Indhold

| Fil / mappe | Beskrivelse |
|---|---|
| `image_sizes.py` | Tjekker billedstørrelser pr. frugt og klasse for at opdage mulige genveje i data |
| `split_data.py` | Stratificeret 70/15/15-split af datasættet til `data/train`, `data/val`, `data/test` |
| `train.py` | Træning med AdamW, early stopping og logging til Weights & Biases |
| `test.py` | Evaluering på testsættet (accuracy, precision, recall, F1), confusion matrix og Grad-CAM |
| `checkpoints/` | De færdigtrænede modeller (bedste epoch efter validation loss) |
| `sample_data/` | 32 testbilleder (2 pr. frugt pr. klasse), så koden kan afprøves uden det fulde datasæt |
| `figures/` | Figurer brugt i rapporten |

## Kør med Docker (anbefalet)

```bash
docker build -t fruit-freshness .
docker run --rm --shm-size=1g -v "$(pwd)/output:/app/figures" fruit-freshness
```

Containeren kører `test.py` med ResNet34-modellen på eksempelbillederne og udskriver metrikker. Confusion matrix og Grad-CAM-billeder havner i mappen `output/`.

`--shm-size=1g` er nødvendig, fordi DataLoaderen bruger flere workers, og Dockers standard for delt hukommelse er for lille til det.

Vil du teste en anden model, så ret `MODEL_NAME` og `PRETRAINED` øverst i `test.py`, og byg imaget igen.

## Kør lokalt uden Docker

```bash
pip install -r requirements.txt
mkdir -p data && cp -r sample_data data/test
python test.py
```

## Træn selv

1. Hent det fulde datasæt *Fresh and Rotten Fruits* (Sultana et al., 2022), og læg de 16 klassemapper i `Original Image/`.
2. `python split_data.py` laver train/val/test-splittet (seed 42).
3. Ret `MODEL_NAME`, `PRETRAINED` og `LEARNING_RATE` i `train.py`, og kør `python train.py`. Kræver en Weights & Biases-konto, eller sæt `WANDB_MODE=offline`.

## Note

Dockerfile, requirements og repo-opsætning er lavet med hjælp fra Claude (AI). Modellering, træning og analyse er mit eget arbejde.

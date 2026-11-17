import torch
import torch.nn as nn #byggeklodserne til neural netværk
from torch.utils.data import DataLoader # DataLoaderen, der fodrer modellen med data: den bestemmer, hvor mange billeder modellen får ad gangen, i hvilken rækkefølge, og sørger for, at de er klar i tide.
from torchvision import datasets, transforms, models
import wandb
import os
import time
# datasets: her bruger du datasets.ImageFolder, som læser dine mapper og automatisk giver billederne label ud fra mappenavnet (fresh = 0, rotten = 1).
# transforms: resize, crop, flip, normalisering osv., altså dine to opskrifter for træning og test.
# models: indeholder ResNet18, 34 og 50, med eller uden de prætrænede ImageNet-vægte.

# =====================================================================
# Denne del kører ALTID (også når test.py importerer fra train.py)
# =====================================================================

# Variabler:
DATA_DIR = "data"
BATCH_SIZE = 32
SEED = 420
NUMBER_EPOCHS = 50
PATIENCE = 7    #


MODEL_NAME = "resnet50"
PRETRAINED = True
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4 # bruges til at straffe store vægte, så netværket lære at generalisere bedre.
device = "cuda" if torch.cuda.is_available() else "cpu" # Spørger om den kan bruge NVIDIA GPU'en, ellers CPU


# Transformer træningsdata:
train_transform = transforms.Compose([
    transforms.Resize(256), # korte side transformeres til 256 pixels
    transforms.RandomCrop(224), #tilfældigt 224x244 udsnit (så den ændre sig pr. epoch lidt)
    transforms.RandomHorizontalFlip(), # tilfældigt spejlvend vandret med 50% sandsynlighed
    transforms.RandomVerticalFlip(), # tilfældigt spejlvend lodret med 50% sandsynlighed
    transforms.ToTensor(), # konverterer billedet til PyTorch-tensorer med nogle små justeringer som passer til pytorch format
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Hvorfor netop de tal? 0,485, 0,456, 0,406 og 0,229, 0,224, 0,225 er gennemsnittet og standardafvigelsen for rød, grøn og blå i hele ImageNet. De prætrænede ResNets har lært på billeder, der var normaliseret med præcis de tal, dermed virker de prætrænede vægte bedre
])

# validering og test data skal ikke have random transformations, da vi vil evaluere modellen på de samme billeder hver gang. Derfor laver vi en anden transform til validering og test:
val_test_transform = transforms.Compose([
    transforms.Resize(256), # korte side transformeres til 256 pixels
    transforms.CenterCrop(224), # tager et 224x224 udsnit fra midten af billedet
    transforms.ToTensor(), # konverterer billedet til PyTorch-tensorer med nogle små justeringer som passer til pytorch format
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# make_model laver en færdig ResNet, der er klar til opgaven
def make_model(name, pretrained = PRETRAINED):   #Den tager to argumenter: name er modellens navn som tekst, fx "resnet18", og pretrained er True eller False.
    weights = "IMAGENET1K_V1" if pretrained else None

    # Hent den rigtige ResNet ud fra name
    if name == "resnet18":
        model = models.resnet18(weights=weights)
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)

    num_features = model.fc.in_features      # Finder ud af, hvor mange tal det sidste lag tager ind. Det er 512 for ResNet18 og 34 og 2048 for ResNet50
    model.fc = nn.Linear(num_features, 1)    # Erstatter det sidste lag (1.000 outputs til ImageNet) med et nyt lag med ét output: en score for, hvor rådden frugten ser ud.

    # Flyt til GPU og returnér
    return model.to(device)



# Denne del kører KUN, når man skriver: python train.py
# (ikke når test.py importerer fra train.py)
if __name__ == "__main__":
    torch.manual_seed(SEED)
    print(f"Bruger: {device}")

    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform = train_transform) # Læs træningsdata fra mappen train, og brug transformen ovenfor
    val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform = val_test_transform) # Læs valideringsdata fra mappen val, og brug transformen ovenfor
    

    print(train_dataset.class_to_idx) # udskriver label mappingen, f.eks. {'fresh': 0, 'rotten': 1}
    print(len(train_dataset), len(val_dataset)) # udskriver antal billeder i træningssættet og valideringssættet

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

    # train_dataset: datasættet, den skal hente billeder fra. Det er dit ImageFolder, som ved, hvor billederne ligger, hvilken label de har, og hvilken transform der skal bruges.
    # batch_size=BATCH_SIZE: hvor mange billeder der samles i hver batch. Med din værdi på 32 bliver de 2.240 træningsbilleder til 70 batches pr. epoch.
    # shuffle=True: blander rækkefølgen på ny i hver epoch, så modellen ikke ser alle friske først og alle rådne bagefter
    # num_workers=4: fire processer på CPU’en åbner og transformerer billeder i baggrunden, mens GPU’en træner på den nuværende batch. Så GPU’en ikke skal vente.

    """
    for images, labels in train_loader: #Går igennem train_loader én batch ad gangen. Hver batch er et par (billeder, labels), som pakkes ud i de to variabler images og labels, ligesom a, b = (5, 10).
        print(images.shape) # Viser formen på billedtensoren. Den skal være torch.Size([32, 3, 224, 224]): 32 billeder, 3 farvekanaler, 224×224 pixels.
        print(labels) # Viser de 32 labels for batchen, 0 er fresh og 1 er rotten.
        break   # stop efter første batch
    """

    model = make_model(MODEL_NAME, pretrained=PRETRAINED)

    kriterie = nn.BCEWithLogitsLoss() # Binary Cross Entropy med Logits. Den kombinerer en Sigmoid-aktivering og en Cross Entropy Loss i ét lag, hvilket er mere stabilt end at have dem som to separate lag.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # AdamW er en optimeringsalgoritme, der justerer vægtene i netværket ud fra gradienterne. Den bruger learning rate og weight decay som parametre.

    run_name = f"{MODEL_NAME}_{'pretrained' if PRETRAINED else 'FRA_BUNDEN'}_lr{LEARNING_RATE}_seed{SEED}"
    os.makedirs("checkpoints", exist_ok=True) 
    #Laver navnet på kørslen ud fra dine indstillinger. Det er en f-string, hvor det, der står i {}, bliver erstattet med en værdi:
    # {MODEL_NAME} bliver til fx resnet18 anden del: Er PRETRAINED sand, bliver det pretrained, ellers FRA BUNDEN.

    # En ordbog (dictionary) med dine indstillinger. Hver linje er et par: til venstre et navn i anførselstegn, til højre værdien fra dine variabler. WandB gemmer dem sammen med kørslen, så du senere kan se præcis, hvad hver model blev trænet med
    wandb.init(
        project="fruit-freshness-resnet",
        name=run_name,
        config={
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUMBER_EPOCHS,
            "seed": SEED,
            "weight_decay": WEIGHT_DECAY,
        },
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(NUMBER_EPOCHS):
        epoch_start = time.time()
        # Træning
        model.train() # sætter modellen i træningstilstand
        running_loss = 0.0
        for images, labels in train_loader: # Går igennem train_loader én batch ad gangen. Hver batch er et par (billeder, labels), som pakkes ud i de to variabler images og labels, ligesom a, b = (5, 10).
            images = images.to(device) # Flytter billedtensoren til GPU'en, hvis den er tilgængelig.
            labels = labels.unsqueeze(1).float().to(device) # gør tre ting ved labels:
            # .float() laver dem fra heltal til decimaltal, fordi loss-funktionen kræver det. unsqueeze(1) ændrer formen fra [32] til [32, 1], så den passer til modellens output. .to(device) flytter dem til GPU’en.

            optimizer.zero_grad() # Nulstiller gradienterne i optimeringsalgoritmen, så de ikke akkumuleres fra tidligere batches (da pytorch som standard lægger gradienter sammen i stedet for at overskrive dem).
            outputs = model(images) # er forward pass: billederne sendes gennem netværket, og du får 32 gæt ud.
            loss = kriterie(outputs, labels) # beregner hvor forkerte gættene er
            loss.backward() # er backward pass: beregner gradienterne for alle vægte i netværket, så de kan justeres.
            optimizer.step() # opdaterer vægtene ud fra gradienterne.

            running_loss += loss.item() # lægger batchens loss til tælleren. .item() tager tallet ud af tensoren, så det bliver et almindeligt Python-tal.

        train_loss = running_loss / len(train_loader)

        # Validering
        model.eval() # sætter modellen i evaluerings tilstand. Batch normalization bruger så de værdier, den har lært under træningen, i stedet for at regne nye ud fra batchen.
        val_loss = 0.0 # Bruges til at akkumulere valideringstab over alle batches i val_loader.
        correct = 0 # Tæller, hvor mange billeder modellen har gættet rigtigt.

        with torch.no_grad():       # slår beregning af gradienter fra for alt, der står indrykket under. Du skal ikke træne på valideringsdata, så der er ingen grund til at regne gradienter. Det gør det hurtigere og bruger mindre hukommelse.
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                outputs = model(images)
                loss = kriterie(outputs, labels)
                val_loss += loss.item()

                probabilities = torch.sigmoid(outputs)          # lav om til sandsynligheder (0-1), bruger altid sigmoid, da vi har et output, der er et tal mellem -inf og +inf. Sigmoid laver det om til et tal mellem 0 og 1.
                predictions = (probabilities > 0.5).float()     # over 50 % = rotten (1), ellers fresh (0)
                correct += (predictions == labels).sum().item() # sammenligner gættene med de rigtige svar. == giver sand eller falsk for hvert billede, .sum() tæller, hvor mange der er rigtige, og .item() laver det til et almindeligt tal.

        val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset) # ividerer antallet af rigtige med antallet af billeder (480), så du får andelen af rigtige.

        print(f"Epoch {epoch+1}/{NUMBER_EPOCHS} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - val acc: {val_acc:.4f}")

        wandb.log({     # sender tallene for denne epoch til WandB. Hver gang den kaldes, bliver det et nyt punkt på kurverne.
            "epoch": epoch + 1, # epoch (selveste loopet) starter på 0, så vi lægger 1 til, så det passer med, hvad vi ser i print
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": time.time() - epoch_start,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"checkpoints/{run_name}_best.pt")
            print(f"  Ny bedste model gemt (val loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping efter epoch {epoch+1}: ingen forbedring i {PATIENCE} epochs")
                break

    total_time = time.time() - start_time
    epochs_run = epoch + 1
    print(f"Træningstid: {total_time/60:.1f} min over {epochs_run} epochs ({total_time/epochs_run:.1f} sek/epoch)")
    wandb.summary["total_time_min"] = total_time / 60
    wandb.summary["sec_per_epoch"] = total_time / epochs_run
    wandb.summary["best_val_loss"] = best_val_loss

    wandb.finish()
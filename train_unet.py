import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from unet_segmentation_dataset import UnetSegmentationDataset
from unetmodel.unet_model import UNet
import matplotlib.pyplot as plt
import os
import sys

def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.item()

def train_unet(outop_dir, epochs):
    # === Settings ===
    EPOCHS = int(epochs)
    BATCH_SIZE = 4
    IMG_SIZE = (512, 256)
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_MODEL_NAME = "unet_teeth_classification.pt"
    OUTPUT_DIR = outop_dir
    patience = 5
    min_delta = 0.0
    counter = 0

    # === Data ===
    train_ds = UnetSegmentationDataset("dataset/images/train", "dataset/masks/train", IMG_SIZE)
    val_ds = UnetSegmentationDataset("dataset/images/val", "dataset/masks/val", IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # === Modell initialization ===
    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # stabiler als BCELoss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"[epoch {epoch+1}/{EPOCHS}]")
        # === Trainingsloop ===
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:

            preds = model(images.to(DEVICE))
            loss = loss_fn(preds, masks.to(DEVICE).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation loop ===
        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                preds = model(images.to(DEVICE))
                loss = loss_fn(preds, masks.to(DEVICE).float())
                val_loss += loss.item()
                dice = dice_score(preds, masks.to(DEVICE).float())
                dice_scores.append(dice)

        avg_dice = sum(dice_scores) / len(dice_scores)
        print(f"🎯 Val Dice: {avg_dice:.4f}")
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        
        # === Early Stopping check ===
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            # === Best Modell save ===
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # model_path = os.path.join(OUTPUT_DIR, f"{epoch+1:02d}_val{avg_val_loss:.4f}.pt")
            model_path = os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME)
            torch.save(model.state_dict(),  model_path)
            print(f"💾 Modell saved as {model_path}")
            counter = 0  # Reset, 
        else:
            counter += 1
            print(f"⏳No enhancement more. Patience Counter: {counter}/{patience}")
            if counter >= patience:
                print("🛑 Loop stopped. Patientce reached (Early Stopping)")
                break




    # === Trainingskurve anzeigen ===
    # Plot Loss-Kurve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss timeline")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
    plt.close()
    print("📊 Loss graphic saved as loss_plot.png")



if __name__=="__main__":
    train_unet(sys.argv[1:][0],sys.argv[1:][1])




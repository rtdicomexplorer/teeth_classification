import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from unet_segmentation_dataset import UnetSegmentationDataset
from unetmodel.unet_model import UNet
import matplotlib.pyplot as plt
import os
import sys
import random
import numpy as np
from tqdm import tqdm

def binary_dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.item()

# === Multiclass Dice Score (optional) ===
def multiclass_dice_score(preds, targets, num_classes=5):
    preds = torch.argmax(preds, dim=1)
    dice_scores = []
    for cls in range(1, num_classes):  # skip background if desired
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dice_scores.append(dice.item())
    return sum(dice_scores) / len(dice_scores)

def train_unet(data_dir, output_dir, epochs, class_items):
    # === Settings ===
    NUM_CLASSES = class_items
    EPOCHS = int(epochs)
    BATCH_SIZE = 4
    IMG_SIZE = None#(256, 512)
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_MODEL_NAME = "best_unet.pt"
    OUTPUT_DIR = output_dir

    print(f"Settings: dataset {dataset_dir}; output_dir {output_dir} nr classes {NUM_CLASSES}; Epochs: {epochs}")


    patience = 5
    min_delta = 0.0
    counter = 0

    images_dir = data_dir # 'dataset_split/'
    pin_memory = False
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        pin_memory = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # === Data ===
    train_ds = UnetSegmentationDataset(os.path.join(images_dir,"images/train"), os.path.join(images_dir,"masks/train"), IMG_SIZE)
    val_ds = UnetSegmentationDataset(os.path.join(images_dir,"images/val"), os.path.join(images_dir,"masks/val"), IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,num_workers=4, pin_memory=pin_memory)

    # === Modell initialization ===
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    #loss_fn = nn.BCEWithLogitsLoss()  # stabiler als BCELoss but just for binary classification not muulticlass 5 in our case
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"[epoch {epoch+1}/{EPOCHS}]")
        # === Trainingsloop ===
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, masks in pbar:

            preds = model(images.to(DEVICE))
            loss = loss_fn(preds, masks.to(DEVICE).long())# Important: integer class labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.refresh()
        pbar.close()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation loop ===
        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}", leave=False)
            for images, masks in pbar:
                preds = model(images.to(DEVICE))
                loss = loss_fn(preds, masks.to(DEVICE).long())
                val_loss += loss.item()
                #dice = binary_dice_score(preds, masks.to(DEVICE).float())
                dice = multiclass_dice_score(preds, masks.to(DEVICE).long(), num_classes=NUM_CLASSES)
                dice_scores.append(dice)
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
                pbar.refresh()
            pbar.close()

        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 🎯 Dice: {avg_dice:.4f}")


        
        # === Early Stopping check ===
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            # === Best Modell save ===
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # model_path = os.path.join(OUTPUT_DIR, f"{epoch+1:02d}_val{avg_val_loss:.4f}.pt")
            model_path = os.path.join(OUTPUT_DIR, OUTPUT_MODEL_NAME)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, model_path)
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
    ''' Input parameter: dataset_dir, output_dir,  epochs, nr of class'''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("class_items", type=int)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    outputdir = args.output_dir #'output_unet'
    epochs = args.epochs #50 #=
    class_items = args.class_items
    train_unet(data_dir=dataset_dir, output_dir=outputdir, epochs=epochs, class_items=class_items)
    #train_unet(sys.argv[1:][0],sys.argv[1:][1])




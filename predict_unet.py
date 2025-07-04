import os
import sys
import torch
import numpy as np
import cv2
from unetmodel import UNet  # Passe den Pfad ggf. an
import albumentations as A
from albumentations.pytorch import ToTensorV2


def predict(model_name,source_dir):
    # === Settings ===
    OUTPUT_DIR = os.path.join(source_dir, "predicted_masks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    IMG_SIZE = (512, 256)  # same size like in Training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # === Transformation (muss exakt wie beim Training sein) ===
    transform = A.Compose([
        A.Resize(IMG_SIZE[1], IMG_SIZE[0]),  # height, width also 256, 512
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_name, map_location=DEVICE))
    model.eval()

   
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"🔍 {len(image_files)} are loading")


    for file in image_files:
        img_path = os.path.join(source_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"⚠️ Error by load: {img_path}")
            continue

        original_size = img.shape[:2]  # (H, W)

        augmented = transform(image=img)
        input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

        with torch.no_grad():
            pred = model(input_tensor)
            pred_mask = (pred > 0.5).float()

        pred_np = pred_mask.squeeze().cpu().numpy()  # (H, W)
        pred_np = (pred_np * 255).astype(np.uint8)
        pred_np = cv2.resize(pred_np, (original_size[1], original_size[0]))

        # mask save
        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_mask.png")
        cv2.imwrite(out_path, pred_np)

        # Overlay creation
        orig_rgb = img.copy()
        mask_color = np.zeros_like(orig_rgb)
        mask_color[:, :, 2] = pred_np  # red
        overlay = cv2.addWeighted(orig_rgb, 0.7, mask_color, 0.3, 0)

        overlay_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        print(f"✅ Mask saved: {out_path}")
        print(f"🎨 Overlay saved: {overlay_path}")

        contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contours on the Overlay (green)
        cv2.drawContours(overlay, contours, -1, color=(0, 255, 0), thickness=2)
        contour_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_contour.png")
        cv2.imwrite(contour_path, overlay)
        print(f"🎨 Contour saved: {contour_path}")

        # === contures as CSV ===
        csv_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_contours.csv")
        with open(csv_path, "w") as f:
            f.write("x,y\n")
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    f.write(f"{x},{y}\n")
                f.write("#\n")  # Separator for the next contur

        print(f"📐 Conture saved: {csv_path}")



if __name__=="__main__":
    ''' Input parameter: model name and source dir to analize'''
    predict(sys.argv[1:][0],sys.argv[1:][1])
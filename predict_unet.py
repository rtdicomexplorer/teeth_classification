import os
import sys
import torch
import numpy as np
import cv2
from unetmodel.unet_model import UNet 
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Optional: define colors for your 5 classes (background, canines, incisors, molars, premolars)
CLASS_COLORS  = {
        0: [0, 0, 0],         # Hintergrund = schwarz
        1: [255, 0, 0],       # Incisors = rot
        2: [0, 255, 0],       # Canines = grün
        3: [0, 0, 255],       # Premolars = blau
        4: [255, 255, 0]      # Molars = yellow
    }


class_names = {
            1: "Incisors",
            2: "Canines",
            3: "Premolars",
            4: "Molars"
        }

def __draw_legend(image, start_x=10, start_y=10, box_size=20, spacing=5, font_scale=0.5, font_thickness=1):
    """
    Draws a legend box on the image showing class color mappings.
    """
    for cls_id, name in class_names.items():
        color = CLASS_COLORS[cls_id]
        label = f"Class {name}"
        top_left = (start_x, start_y + cls_id * (box_size + spacing))
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
        # Draw color box
        cv2.rectangle(image, top_left, bottom_right, color, -1)

        # Put text next to box
        text_pos = (bottom_right[0] + 5, bottom_right[1] - 5)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def predict(model_path, source_dir):
    # === Settings ===
    OUTPUT_DIR = os.path.join(source_dir, "predicted_masks")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    IMG_SIZE = (256, 512)  # same size like in Training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]), 
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

    ##binary mode 1 channel
    # model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE))


    # === Load Multi-class Model ===
    model = UNet(in_channels=3, out_channels=5).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)  # ✅ Correct
    
    model.eval()

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"🔍 Found {len(image_files)} images")

    for file in image_files:
        img_path = os.path.join(source_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        original_size = img.shape[:2]  # (H, W)
        augmented = transform(image=img)
        input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

        with torch.no_grad():
            output = model(input_tensor)  # (1, 5, H, W)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # Resize prediction back to original image size
        pred_resized = cv2.resize(pred.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Save mask image (grayscale or pseudo-color)
        color_mask = np.zeros_like(img)
        for cls, color in CLASS_COLORS.items():
            color_mask[pred_resized == cls] = color

        # === Save plain color mask ===
        mask_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_mask.png")
        cv2.imwrite(mask_path, color_mask)

        # === Overlay ===
        overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

        # 🔠 Draw legend on overlay
        __draw_legend(overlay)

        overlay_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        # === Contours per class (excluding background) ===
        for cls in range(1, 5):
            cls_mask = np.uint8(pred_resized == cls)
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_img = overlay.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

            contour_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_class{cls}_contour.png")
            cv2.imwrite(contour_path, contour_img)

            # === Save contours to CSV ===
            csv_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_class{cls}_contours.csv")
            with open(csv_path, "w") as f:
                f.write("x,y\n")
                for contour in contours:
                    for point in contour:
                        x, y = point[0]
                        f.write(f"{x},{y}\n")
                    f.write("#\n")

        print(f"✅ Processed: {file}")
    print(f"👋🦷📊 Predictions completed")



if __name__=="__main__":
    ''' Input parameter: model name and source dir to analize'''

    model = './output_unet/unet_teeth_classification.pt'
    
    images = './panoramic_x_rays/'
    # predict(model,images)
    predict(sys.argv[1:][0],sys.argv[1:][1])
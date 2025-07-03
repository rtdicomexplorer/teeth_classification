import torch
from ultralytics import YOLO
import os
import sys




# | Parameter   | Bedeutung                                  | Empfehlung                                                       |
# | ----------- | ------------------------------------------ | ---------------------------------------------------------------- |
# | `data`      | Pfad zur `.yaml`-Datei mit deinen Daten    | `"karies.yaml"`                                                  |
# | `epochs`    | Anzahl der Trainingsdurchläufe             | z. B. `50`, je mehr = besser (bis Overfitting)                   |
# | `imgsz`     | Bildgröße (quadratisch)                    | `640` ist Standard; `512`, `768`, `1024` bei mehr Rechenleistung |
# | `batch`     | Batch-Größe (Bilder pro Schritt)           | Hängt von RAM/GPU ab. `8` ist gut für Einsteiger                 |
# | `device`    | `"0"` = erste GPU, `"cpu"` = CPU verwenden | `"0"` (GPU), `"cpu"` nur wenn nötig                              |
# | `optimizer` | Trainingsalgorithmus                       | `'SGD'` (gut für YOLO), `'Adam'` ist möglich                     |
# | `lr0`       | Start-Lernrate                             | Standard ist `0.01`; kannst später feintunen                     |
# | `workers`   | Datenlader-Threads                         | `8` ist oft optimal, weniger bei schwacher CPU                   |
# | `project`   | Ordner für Ausgaben                        | z. B. `"runs/train"`                                             |
# | `name`      | Unterordnername für das Training           | z. B. `"karies"`                                                 |
# | `exist_ok`  | Verzeichnis überschreiben?                 | `True` = ja, `False` = Fehler wenn Ordner schon da               |


#results:

# | Datei/Ordner           | Beschreibung                                                                |
# | ---------------------- | --------------------------------------------------------------------------- |
# | `weights/best.pt`      | ✅ **Bestes Modell** basierend auf Validierungs-mAP (empfohlen für Inferenz) |
# | `weights/last.pt`      | Letzter Checkpoint des Trainings                                            |
# | `results.csv`          | Trainingsverlauf als CSV (Losses, mAP, Precision etc. pro Epoch)            |
# | `results.png`          | Grafik der Trainingsverläufe (Loss, mAP usw.)                               |
# | `confusion_matrix.png` | Matrix der Vorhersagegenauigkeit je Klasse                                  |
# | `P*curve.png`          | Precision/Recall-Kurven                                                     |
# | `train_batch*.jpg`     | Visualisierung einzelner Trainingsbatches                                   |
# | `val_batch*.jpg`       | Visualisierung der Validierungsvorhersagen                                  |


def train_yolo_model(output_folder, epochs):
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device selected: {device}")

    model = YOLO("yolov8m.pt") # yolov8n.pt

    model.train(
        data="dataset/data.yaml",
        epochs=int(epochs),
        imgsz=400,
        batch=84,
        # lr0=0.01,
        # optimizer='SGD',
        # workers=8,
        device=device,
        project='runs/train',
        name='teeth-classification',
        exist_ok=True
    )


if __name__=='__main__':
  train_yolo_model(sys.argv[1:][0],sys.argv[1:][1])
  
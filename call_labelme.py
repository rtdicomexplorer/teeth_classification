import subprocess
import onnxruntime
print(onnxruntime.__version__)

# Open a specific image
#subprocess.run(["labelme", "path/to/image.jpg"])

# Or open a directory of images
subprocess.run(["labelme", "dataset/images/val"])

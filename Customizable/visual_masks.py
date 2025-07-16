import cv2
import os
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from detectron2.data import MetadataCatalog, DatasetCatalog

# Define Variables
model_weights="./model_1class.pth"
image_path=""
device="cpu"

cfg = get_cfg()
cfg.OUTPUT_DIR = "./"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE=device
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_weights)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold

predictor = DefaultPredictor(cfg)



# === Inference and Visualization ===
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run prediction
outputs = predictor(image)

# Visualize the predictions
v = Visualizer(image_rgb, MetadataCatalog.get("bubble_dataset"), scale=1.0)

output_image = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

# Plot using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis("off")
plt.title("Predicted Masks")
plt.show()


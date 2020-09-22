import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

from ssd.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from ssd.keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd.keras_loss_function.keras_ssd_loss import SSDLoss
from utils.logger import logger
from base import BaseDetector

IMG_HEIGHT = 512
IMG_WIDTH = 512


class SSDDetector(BaseDetector):
    def find_heads(self, img_path: str, cfg: dict) -> []:
        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

        # Clear previous models from memory.
        K.clear_session()

        decode_layer = DecodeDetections(img_height=IMG_HEIGHT,
                                        img_width=IMG_WIDTH,
                                        confidence_thresh=cfg["confidence_threshold"],
                                        iou_threshold=cfg["iou_threshold"],
                                        top_k=cfg["top_k"],
                                        nms_max_output_size=cfg["nms_max_output_size"])

        model = load_model(cfg["weights"], custom_objects={'AnchorBoxes': AnchorBoxes,
                                                         'L2Normalization': L2Normalization,
                                                         'DecodeDetections': decode_layer,
                                                         'compute_loss': ssd_loss.compute_loss})
        orig_images = []
        input_images = []

        orig_images.append(image.load_img(img_path))
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)

        y_pred = model.predict(input_images)

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > cfg["confidence_threshold"]] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        logger.info("Predicted boxes:\n")
        logger.info('   class   conf xmin   ymin   xmax   ymax')
        logger.info(y_pred_thresh[0])
        logger.info(f"Found {len(y_pred_thresh[0])} objects")
        result = []

        for box in y_pred_thresh[0]:
            xmin = int(box[2] * np.array(orig_images[0]).shape[1] / IMG_WIDTH)
            ymin = int(box[3] * np.array(orig_images[0]).shape[0] / IMG_HEIGHT)
            xmax = int(box[4] * np.array(orig_images[0]).shape[1] / IMG_WIDTH)
            ymax = int(box[5] * np.array(orig_images[0]).shape[0] / IMG_HEIGHT)
            res = [xmin, ymin, xmax, ymax]
            result.append([res])
        return result

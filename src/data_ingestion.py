from logger.log_config import setup_logger
from utils.config import load_params
import os
import cv2
import numpy as np

params = load_params()
logger = setup_logger()

logger.info("pipeline started")

def data_loading(path=params["data_ingestion"]["dataset_path"]):
    
    images = []
    labels = []
    
    try:

        authentic_path = os.path.join(path, "Au")
        logger.info(f"Extracting authentic path: {authentic_path}")
        
        for img_file in os.listdir(authentic_path):
            if img_file.endswith((".jpg", ".png", ".tif")):
                img_path = os.path.join(authentic_path, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(0)
                    logger.debug(f"Loaded authentic: {img_file}")
                else:
                    logger.warning(f"Could not load: {img_file}")

        logger.info(f"Total authentic images: {len(images)}")

        tampered_path = os.path.join(path, "Tp")
        logger.info(f"Loading tampered images from: {tampered_path}")
        
        for img_file in os.listdir(tampered_path):
            if img_file.endswith((".jpg", ".png", ".tif")):
                img_path = os.path.join(tampered_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(1)
                    logger.debug(f"Loaded tampered: {img_file}")
                else:
                    logger.warning(f"Could not load: {img_file}")
        
        logger.info(f"Total tampered images: {len(images)}")
        logger.info(f"Total dataset size: {len(images)}")

        images = np.array(images)
        labels = np.array(labels)

        output_path = params["artifacts"]["raw_data"]
        os.makedirs(output_path, exist_ok=True)

        np.save(os.path.join(output_path, "images.npy"), images)
        np.save(os.path.join(output_path, "labels.npy"), labels)

        logger.info(f"Artifacts saved to: {output_path}")
        logger.info(f"images.npy shape: {images.shape}")
        logger.info(f"labels.npy shape: {labels.shape}")
        
        return images, labels

    except Exception as e:
        logger.error(f"Error occurred during data ingestion: {e}")
        return None, None


if __name__ == "__main__":
    images, labels = data_loading()
    if images is not None:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Authentic: {sum(labels == 0)}")
        print(f"Tampered:  {sum(labels == 1)}")
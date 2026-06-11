import numpy as np
from logger.log_config import setup_logger
from utils.config import load_params
import os
import cv2


def preprocess():
    params = load_params()
    logger = setup_logger()

    logger.info("starting preprocessing of the data")

    try:
        images_path = os.path.join(params["artifacts"]["raw_data"], "images.npy")
        labels_path = os.path.join(params["artifacts"]["raw_data"], "labels.npy")

        images = np.load(images_path)
        labels = np.load(labels_path)

        logger.info(f"images loaded, shape: {images.shape}") 
        gray_images = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray)

        gray_images = np.array(gray_images)
        logger.info(f"converted to grayscale, shape: {gray_images.shape}")  


        gray_images = np.expand_dims(gray_images, axis=-1)
        logger.info(f"channel dimension added, shape: {gray_images.shape}") 

        normalized_images = gray_images.astype('float32') / 255.0
        logger.info("images normalized to [0,1]")

        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            normalized_images, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
        logger.info(f"Train labels distribution: {np.bincount(y_train)}")
        logger.info(f"Val labels distribution: {np.bincount(y_val)}")

        output_path = params["artifacts"]["processed_data"]
        os.makedirs(output_path, exist_ok=True)

        np.save(os.path.join(output_path, "X_train.npy"), X_train)
        np.save(os.path.join(output_path, "X_val.npy"), X_val)
        np.save(os.path.join(output_path, "y_train.npy"), y_train)
        np.save(os.path.join(output_path, "y_val.npy"), y_val)

        logger.info(f"preprocessed data saved to: {output_path}")

    except Exception as e:
        logger.error(f"error occurred during preprocessing: {e}")


if __name__ == "__main__":
    preprocess()
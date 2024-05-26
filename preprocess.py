import numpy as np
import os
from multiprocessing import Pool, cpu_count
import gc
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_single_image(args):
    img_path, target_size = args
    img = image.load_img(os.path.join("images", img_path), target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_images(image_paths, target_size=(220, 220)):
    with Pool(cpu_count()) as pool:
        img_list = list(tqdm(pool.imap(
            preprocess_single_image, [(img_path, target_size) for img_path in image_paths]), total=len(image_paths)))
    return np.vstack(img_list)

if __name__ == "__main__":
    image_paths = os.listdir("images/")
    images = preprocess_images(image_paths)
    np.savez("preprocessed_features.npz", images)
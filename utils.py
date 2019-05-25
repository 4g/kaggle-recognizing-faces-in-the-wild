import glob
from tqdm import tqdm
import cv2
from keras.preprocessing import image as imglib
import numpy as np

train_images_path = "data/train/*"
test_images_path = "data/test/*.jpg"
train_pairs_path = "data/train_relationships.csv"
submission_pairs_path = "data/sample_submission.csv"


def load_data():
    _train_images = []
    _train_labels = []

    for family in tqdm(glob.glob(train_images_path)):
        for members in glob.glob(family + "/*"):
            for image_path in glob.glob(members + "/*.jpg"):
                image = imglib.load_img(image_path)
                image = imglib.img_to_array(image, dtype=np.float32)
                _train_images.append(image)
                label = "/".join(members.split("/")[-2:])
                _train_labels.append(label)

    _test_images = []
    _test_labels = []
    for image_path in tqdm(glob.glob(test_images_path)):
        image = imglib.load_img(image_path)
        image = imglib.img_to_array(image, dtype=np.float32)
        _test_images.append(image)
        label = "/".join(image_path.split("/")[-1:])

        _test_labels.append(label)

    _train_pairs = [tuple(i.strip().split(",")) for i in open(train_pairs_path)][1:]
    submission_pairs = [tuple(i.strip().split(",")[0].split("-")) for i in open(submission_pairs_path)][1:]

    label_counts = {}

    for label in _train_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    pair_count = 0
    for l1, l2 in _train_pairs:
        pair_count += label_counts.get(l1, 0) * label_counts.get(l2, 0)

    print("Num image pairs", pair_count)
    print("Num train images", len(_train_images))
    print("Num test images", len(_test_images))
    print("Num person pairs", len(_train_pairs))
    print("Num submission pairs", len(submission_pairs))

    return _train_images, _train_labels, _test_images, _test_labels, _train_pairs, submission_pairs

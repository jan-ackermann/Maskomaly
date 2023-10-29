import os

import cv2
import numpy as np

from torch.utils.data import Dataset

class RoadAnomaly(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path))
        self.images_paths = [os.path.join(root_path, file) for file in self.images_paths if file.endswith(".jpg")]
        self.images_paths.sort()

        self.labels_paths = os.listdir(os.path.join(root_path))
        self.labels_paths = [os.path.join(root_path, direc, "labels_semantic_color.png") for
                             direc in self.labels_paths if os.path.isdir(os.path.join(root_path, direc))]
        self.labels_paths.sort()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 1]
        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 69] = 1

        ignore = np.zeros_like(label, dtype=np.uint8)
        #ignore[label == 100] = 1

        return image, anomaly_gt, ignore, os.path.basename(image_path)
class SMIYC_FULL(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path, "images"))
        self.images_paths.sort()
        self.images_paths = [os.path.join(root_path, "images", image) for image in self.images_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        return image, np.zeros_like(image[:,:,0]), np.zeros_like(image[:,:,0]), os.path.basename(image_path)

class SMIYC(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path, "images"))
        self.images_paths.sort()
        self.images_paths = [os.path.join(root_path, "images", image) for image in self.images_paths
                             if image.startswith("validation")]

        self.labels_paths = os.listdir(os.path.join(root_path, "labels_masks"))
        self.labels_paths.sort()
        self.labels_paths = [os.path.join(root_path, "labels_masks", label) for label in self.labels_paths
                             if label.endswith("color.png")]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 1]

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 102] = 1

        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 0] = 1

        return image, anomaly_gt, ignore, os.path.basename(image_path)

class FishyScapesStatic(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path, "images"))
        self.images_paths.sort()
        self.images_paths = [os.path.join(root_path, "images", image) for image in self.images_paths]

        self.labels_paths = os.listdir(os.path.join(root_path, "labels"))
        self.labels_paths.sort()
        self.labels_paths = [os.path.join(root_path, "labels", label) for label in self.labels_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 0]

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 1] = 1

        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 255] = 1

        return image, anomaly_gt, ignore, os.path.basename(image_path)

class FishyScapes(Dataset):
    def __init__(self, root_path):
        self.images_paths = os.listdir(os.path.join(root_path, "images"))
        self.images_paths.sort()
        self.images_paths = [os.path.join(root_path, "images", image) for image in self.images_paths]

        self.labels_paths = os.listdir(os.path.join(root_path, "labels"))
        self.labels_paths.sort()
        self.labels_paths = [os.path.join(root_path, "labels", label) for label in self.labels_paths]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 0]

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 120] = 1

        ignore = np.zeros_like(label, dtype=np.uint8)
        ignore[label == 255] = 1

        return image, anomaly_gt, ignore, os.path.basename(image_path)

class StreetHazards(Dataset):
    def __init__(self, root_path):
        images_path_t5 = os.path.join(root_path, "test", "images", "test", "t5")
        images_path_t6 = os.path.join(root_path, "test", "images", "test", "t6")

        self.images_paths = [os.path.join(images_path_t5, file) for file in os.listdir(images_path_t5)]
        self.images_paths = self.images_paths + [os.path.join(images_path_t6, file) for file in
                                                 os.listdir(images_path_t6)]
        self.images_paths.sort()

        labels_path_t5 = os.path.join(root_path, "test", "annotations", "test", "t5")
        labels_path_t6 = os.path.join(root_path, "test", "annotations", "test", "t6")

        self.labels_paths = [os.path.join(labels_path_t5, file) for file in
                             os.listdir(labels_path_t5)]
        self.labels_paths = self.labels_paths + [os.path.join(labels_path_t6, file) for file in
                                                 os.listdir(labels_path_t6)]
        self.labels_paths.sort()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)

        label_path = self.labels_paths[idx]
        label = cv2.imread(label_path)
        label = label[:, :, 0]

        anomaly_gt = np.zeros_like(label, dtype=np.uint8)
        anomaly_gt[label == 255] = 1

        ignore = np.zeros_like(label, dtype=np.uint8)

        return image, anomaly_gt, ignore, f'{os.path.split(image_path)[-2]}_{os.path.basename(image_path)}'

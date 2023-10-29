import argparse

import cv2
import numpy as np
import os
import sys
import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

from mask2former import add_maskformer2_config

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input",
        help="path to input data directory",
    )
    args.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    args.add_argument("--output", default="smiyc")
    args.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return args

T_MASK = 0.9

def get_smiyc(input_dir):
    gt_addon = "labels_masks"
    data_addon = "images"
    data_file_paths = [os.path.join(input_dir, data_addon, file) for file in
                       os.listdir(os.path.join(input_dir, data_addon)) if file.startswith("validation")]
    gt_file_paths = [os.path.join(input_dir, gt_addon, file) for file in os.listdir(os.path.join(input_dir, gt_addon))
                     if file.endswith("semantic.png")]
    data_file_paths.sort()
    gt_file_paths.sort()
    return data_file_paths, gt_file_paths


def get_predictions(model, images, gt):
    counter_outlier = np.zeros((100))
    counter_inlier = np.zeros((100))
    counter_predicted = np.zeros((len(images), 100), dtype=np.float32)

    i = 0
    for image, label in tqdm.tqdm(zip(images, gt)):

        image = cv2.imread(image)
        label = cv2.imread(label)
        label = label[:, :, 0]

        ignore = label == 255
        ignore_count = np.count_nonzero(ignore)

        label = label == 1
        _, _, mask_results = model(image)
        mask_results = mask_results.sigmoid().cpu().numpy()

        total_anom = np.count_nonzero(label)
        label_stacked = np.asarray([label] * mask_results.shape[0])

        results = np.zeros(label_stacked.shape, dtype=np.float32)

        for j in range(100):
            results[j] = cv2.resize(mask_results[j], image.shape[:2][::-1], interpolation=cv2.INTER_AREA)

        mask_results = results

        mask_results = (mask_results > T_MASK).astype(np.float16)

        void_pred = np.logical_and(mask_results, label_stacked)
        class_pred = np.logical_and(mask_results, 1 - label_stacked)

        void_counts = np.count_nonzero(void_pred, axis=(1, 2)).astype(np.float32)
        class_counts = np.count_nonzero(class_pred, axis=(1, 2)).astype(np.float32)

        non_ignore_mask_count = np.count_nonzero(mask_results, axis=(1, 2)).astype(np.float32) - ignore_count

        counter_outlier += void_counts
        counter_inlier += class_counts
        counter_predicted[i] = void_counts / (non_ignore_mask_count + total_anom - void_counts)
        i += 1

    return counter_inlier, counter_outlier, counter_predicted


if __name__ == '__main__':
    args = get_args().parse_args()

    data_paths, gt_paths = get_smiyc(args.input)

    cfg = setup_cfg(args)
    model = DefaultPredictor(cfg)

    count_inlier, count_outlier, counter_predicted = get_predictions(model, data_paths, gt_paths)

    cp = counter_predicted.mean(axis=0)
    tp = count_inlier / (count_outlier + count_inlier)

    order = np.argsort(cp)

    np.savez_compressed("results_fs.npz", tp=tp, cp=cp)

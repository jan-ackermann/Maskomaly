import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from torch.nn import functional as F

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


class BaseModel():
    def __init__(self, args):
        cfg = setup_cfg(args)
        self.model = DefaultPredictor(cfg)
        self.times = []

    def get_soft_mask(self, image):
        raise Exception("Needs to be overloaded!")

    def get_time(self):
        return sum(self.times) / len(self.times)


class BaseSegmentationModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

    def get_probs_and_seg(self, image):
        segmentation, mask_cls_result, mask_pred_result = self.model(image)
        mask_cls_result = F.softmax(mask_cls_result, dim=1).cpu().numpy()
        mask_pred_result = mask_pred_result.sigmoid().cpu().numpy()

        return mask_cls_result, mask_pred_result, segmentation

class Maskomaly(BaseSegmentationModel):
    def __init__(self, args):
        super().__init__(args)
        if args.analysis_file:
            self.cp = np.load(args.analysis_file)["cp"]
            self.ranking = np.argsort(self.cp)[::-1]
            self.cp = self.cp[self.ranking]
            self.take = int(args.masks) | np.argmax(self.cp < 0.25)
            self.ranking = self.ranking[:self.take]
        else:
            self.ranking = [49, 31, 83, 32]

        self.class_stats = np.zeros(10)
        self.pred_stats = np.zeros(10)

    def get_soft_mask(self, image):
        mask_cls_result, mask_pred_result, _ = self.get_probs_and_seg(image)

        start_t = time.time()
        soft_mask = np.ones_like(mask_pred_result[0], dtype=np.float32)

        # create the anomalous mask based prediction
        soft_mask2 = np.zeros_like(mask_pred_result[0], dtype=np.float32)
        for ind in [49, 31, 83, 32]: # masks as ranked by importance for predicting anomalies by SMIYC validation
            self.class_stats[int(np.max(mask_cls_result[ind]) * 10)] += 1
            for t in range(10):
                self.pred_stats[t] += np.count_nonzero(mask_pred_result[ind] > t/10)
            soft_mask2 = np.maximum(soft_mask2, mask_pred_result[ind] * np.max(mask_cls_result[ind]))

        # create the rejection based prediction
        for i in range(mask_cls_result.shape[0]):
            maximum_index = np.argmax(mask_cls_result[i])
            if maximum_index != 19 and mask_cls_result[i][maximum_index] > 0.7:

                for t in range(10):
                    self.pred_stats[t] += np.count_nonzero(mask_pred_result[i] > t/10)
                self.class_stats[int(np.max(mask_cls_result[i]) * 10)] += 1
                neg = 1 - mask_pred_result[i] * mask_cls_result[i][maximum_index]
                soft_mask = np.minimum(soft_mask, neg)

        # compute the indices of the classes that are non-void predictions
        max_indices = np.argmax(mask_cls_result, axis=1)
        positive = mask_pred_result[max_indices != 19.0]

        # remove the borders of the non-void predictions up to epsilon=0.001
        for i in range(positive.shape[0]):
            for j in range(i + 1, positive.shape[0]):
                neg_border = np.clip((1 - np.logical_and(positive[i] > 0.1, positive[j] > 0.1)) + 0.0, 0, 1)
                soft_mask = np.minimum(neg_border, soft_mask)

        # fix the mislabeling of ground
        for i in [19, 24]: # masks as ranked by importance for predicting ground by Cityscapes validation
            maximum_index = np.argmax(mask_cls_result[i])
            neg = 1 - mask_pred_result[i] * mask_cls_result[i][maximum_index]
            soft_mask = np.minimum(soft_mask, neg)

        # interpolate the masks with lambda = 0.6, any lambda > 0.5 is fine!
        soft_mask = 0.6 * soft_mask + 0.4 * soft_mask2

        # fix if mask is not the same size due to Mask2Former
        soft_mask = cv2.resize(soft_mask, image.shape[:2][::-1], interpolation=cv2.INTER_AREA)
        self.times.append(time.time() - start_t)
        return soft_mask

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve, f1_score
from torch.nn import functional as F
import tqdm


def calculate_auroc(gt, conf):
    fpr, tpr, threshold = roc_curve(gt, conf)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    for i, j, k in zip(tpr, fpr, threshold):
        if i > 0.9:
            fpr_best = j
            break
    return roc_auc, fpr_best, k


def get_best_f1(anomaly, ignore, gt, n=100):
    anomaly = anomaly[ignore < 1]
    gt = gt[ignore < 1]
    best = 0
    thresh = 0
    for i in range(n):
        temp = anomaly.copy()
        temp[temp < i / n] = 0
        temp[temp > 0] = 1
        result = f1_score(gt.flatten(), temp.flatten())
        if result > best:
            best = result
            thresh = i / n
    return thresh


def calculate_aupr(gt, conf):
    precision, recall, t = precision_recall_curve(gt, conf)
    s = recall.argsort()
    precision = precision[s]
    recall = recall[s]
    return auc(recall, precision)


def get_probs_from_masks(mask_cls_results, mask_pred_results):
    """
    params:
        mask_cls_results (np.ndarray): with shape #masks x #classes
    """

    anomaly_mask = np.ones_like(mask_pred_results[0], dtype=np.float32)
    for i in range(mask_pred_results.shape[0]):
        maximum_index = np.argmax(mask_cls_results[i])
        if maximum_index != 19 and mask_cls_results[i][maximum_index] > 0.75:
            neg = 1 - mask_pred_results[i] * mask_cls_results[i][maximum_index]
            anomaly_mask = np.minimum(anomaly_mask, neg)
    anomaly_mask = np.maximum(anomaly_mask, 0)
    return anomaly_mask


def write_anomaly_to_image(img, gt, anomaly_probs, ignore, path, n=100):
    thresh = get_best_f1(anomaly_probs, ignore, gt, n)
    img[:, :, 0][np.logical_and(anomaly_probs > thresh, ignore < 1)] = 0
    img[:, :, 1][np.logical_and(anomaly_probs > thresh, ignore < 1)] = 0
    cv2.imwrite(path + "_anomaly.png", img)

def write_anomaly_to_image_no_gt(img, anomaly_probs, path, n=10):
    for i in range(n):
        test = img.copy()
        thresh = i/n
        test[:, :, 0][anomaly_probs > thresh] = 0
        test[:, :, 1][anomaly_probs > thresh] = 0
        cv2.imwrite(path + f'_thresh_{i:02d}.png', test)



def write_masks(img, masks, root_path):
    os.makedirs(root_path, exist_ok=True)
    current = 0
    for mask in masks:
        write_anomaly_to_image(img, mask, os.path.join(root_path, f'{current:03d}.png'))


def report_results(ap, auroc, fpr, aupr, plot_graph=False, y_true=None, y_probs=None):
    print(f'The average precision is: {100 * ap:.2f}%')
    print(f'The auroc is: {100 * auroc:.2f}%')
    print(f'The fpr is: {100 * fpr:.2f}%')
    print(f'The aupr is: {100 * aupr:.2f}%')
    if plot_graph:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.plot(fpr, tpr, label="auroc")
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.plot(precision, recall, label="aupr")
        plt.legend(loc=4)
        plt.show()


def get_mdm_f1_metric(gt, pred, thresh=0.7, approximation=1000):
    """Computes the MDM-F1 metric.
    
    Takes long to compute when the approximation rate is high!
    """

    n = approximation

    results = np.zeros(n)

    for i in range(n):
        print(i)
        temp = pred.copy()
        temp[temp < i / n] = 0
        temp[temp > 0] = 1
        result = f1_score(gt.flatten(), temp.flatten())

        results[i] = result

    best = 0

    for i in range(results.shape[0]):
        current = 0
        for j in range(i, results.shape[0]):
            if results[j] > thresh:
                current += 1
            else:
                best = max(best,current)
                break
        best = max(best, current)
    
    return best/n

def get_scores(ground_truths, anomaly_probs, ignores, mode="total"):
    """
    params:
        ground_truths (np.ndarray): with shape #images x width x height of type np.uint8
            1 indicates that an anomaly is present at that pixel and 0 if there is none
        anomaly_probs (np.ndarray): with shape #images x width x height of type np.float32
        ignores (np.ndarray): with shape #images x width x height of type np.uint8
            1 indicates that the pixel should not be used for evaluation and 0 otherwise
        mode (string): of [total, image] that decides whether the evaluation should be done per image or in total

    returns:
        average_precision (np.float32)
        roc_auc (np.float32)
        fpr (np.float32)
    """
    # make sure that the mode is one of the two
    possible_modes = ["total", "image"]
    if mode not in possible_modes:
        raise Exception("The mode can only be one of total or image.")

    if mode == possible_modes[0]:
        # Remove the pixel values from ignores for evaluation automatically flattens the array
        ground_truths = ground_truths[ignores < 1]
        anomaly_probs = anomaly_probs[ignores < 1]

        average_precision = average_precision_score(ground_truths, anomaly_probs)
        roc_auc, fpr, t = calculate_auroc(ground_truths, anomaly_probs)
        aupr = calculate_aupr(ground_truths, anomaly_probs)

        return average_precision, roc_auc, fpr, aupr
    else:
        average_precision_values = []
        roc_auc_values = []
        fpr_values = []
        aupr_values = []
        for i in range(ground_truths.shape[0]):

            ground_truth = ground_truths[i]
            anomaly_prob = anomaly_probs[i]


            ground_truth = ground_truth.flatten()
            anomaly_prob = anomaly_prob.flatten()

            average_precision = average_precision_score(ground_truth, anomaly_prob)
            average_precision_values.append(average_precision)
            roc_auc, fpr, t = calculate_auroc(ground_truth, anomaly_prob)
            aupr = calculate_aupr(ground_truth, anomaly_prob)
            roc_auc_values.append(roc_auc)
            fpr_values.append(fpr)
            aupr_values.append(aupr)

        # convert arrays to numpy arrays
        average_precision_values = np.asarray(average_precision_values, dtype=np.float32)
        roc_auc_values = np.asarray(roc_auc_values, dtype=np.float32)
        fpr_values = np.asarray(fpr_values, dtype=np.float32)
        aupr_values = np.asarray(aupr_values, dtype=np.float32)

        return np.nanmean(average_precision_values), np.nanmean(roc_auc_values), np.nanmean(fpr_values), np.nanmean(aupr_values)

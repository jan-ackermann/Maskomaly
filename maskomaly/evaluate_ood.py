import argparse
import os

import numpy as np
import tqdm

from eval import get_scores, write_anomaly_to_image, write_anomaly_to_image_no_gt
import cv2


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input",
        help="path to input data"
    )
    parser.add_argument(
        "--dataset",
        help="Choose the dataset you wish to evaluate on.",
        default="smiyc_anomaly"
    )
    parser.add_argument(
        "--output",
        help="Choose the output destination."
    )
    parser.add_argument(
        "--logging_interval",
        help="How often should the anomaly be drawn?",
        default=1
    )

    return parser


def heatmap(image, probs):
    heatmap_img = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)
    return super_imposed_img


def main():
    args = get_parser().parse_args()

    gt_values = []
    ignore_values = []
    soft_mask_values = []

    os.makedirs(args.output, exist_ok=True)
    files = os.listdir(args.input)
    files.sort()

    for i, path in enumerate(tqdm.tqdm(files)):
        data = np.load(os.path.join(args.input, path))
        gt_values.append(data["gt"])
        ignore_values.append(data["ignore"])
        soft_mask = data["soft_mask"]
        soft_mask = cv2.resize(soft_mask, data["gt"].shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        soft_mask_values.append(soft_mask)

        h = heatmap(data["image"].copy(), soft_mask)
        cv2.imwrite(os.path.join(args.output, f'{i:03d}_heat.png'), h)
        
        

        if i % args.logging_interval == 0 and args.datset != 'smiyc_full':
            write_anomaly_to_image(data["image"], data["gt"], soft_mask, data["ignore"], os.path.join(args.output, f'{i:03d}'), 100)

    mode = "total"
    if args.dataset == "sh":
        mode = "image"

    ap, roc, fpr, aupr = get_scores(np.asarray(gt_values), np.asarray(soft_mask_values), np.asarray(ignore_values),
                                    mode=mode)
    with open(os.path.join(args.output, "results.txt"), "w") as file:
        file.write(f'{100 * ap:.3f}%\n')
        file.write(f'{100 * roc:.2f}%\n')
        file.write(f'{100 * fpr:.2f}%\n')
        file.write(f'{100 * aupr:.2f}%\n')
        file.write(f'The average precision is: {100 * ap:.3f}%\n')
        file.write(f'The auroc is: {100 * roc:.2f}%\n')
        file.write(f'The fpr is: {100 * fpr:.2f}%\n')
        file.write(f'The aupr is: {100 * aupr:.2f}%\n')


if __name__ == "__main__":
    main()

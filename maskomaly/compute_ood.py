import argparse
import multiprocessing as mp
import os

import numpy as np
import tqdm
from torch.utils.data import DataLoader

from datasets import SMIYC, FishyScapes, StreetHazards, SMIYC_FULL, FishyScapesStatic, RoadAnomaly
from model import Maskomaly


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="path to input data"
    )
    parser.add_argument(
        "--dataset",
        help="Choose the dataset you wish to evaluate on.",
    )
    parser.add_argument(
        "--output",
        help="Choose the output destination.",
    )
    parser.add_argument(
        "--masks",
        help="Number of masks to use",
        default=4,
    )
    parser.add_argument(
        "--analysis_file",
        help="Path to the file produced by the IoU overlap finder. Not necessary if one wants to use the precomputed query indices.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser

def choose_dataset(args):
    if args.dataset == "smiyc_full":
        dataset = SMIYC_FULL(args.input)
    elif args.dataset.startswith("smiyc"):
        dataset = SMIYC(args.input)
    elif args.dataset == "road_anomaly":
        dataset = RoadAnomaly(args.input)
    elif args.dataset == "fs_static":
        dataset = FishyScapesStatic(args.input)
    elif args.dataset.startswith("fs"):
        dataset = FishyScapes(args.input)
    elif args.dataset.startswith("sh"):
        dataset = StreetHazards(args.input)
    else:
        raise Exception("Dataset not found!")
    return dataset


def choose_model(args):
    return Maskomaly(args)


def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    dataset = choose_dataset(args)
    dataloader = DataLoader(dataset)

    model = choose_model(args)

    os.makedirs(args.output, exist_ok=True)

    for image, gt, ignore, file in tqdm.tqdm(dataloader):
        image = image.cpu().numpy()[0]
        gt = gt.cpu().numpy()[0]
        ignore = ignore.cpu().numpy()[0]
        soft_mask = model.get_soft_mask(image)
        np.savez(os.path.join(args.output, file[0]), image=image, gt=gt, ignore=ignore, soft_mask=soft_mask)

    print(sum(model.times)/len(model.times))
    print(model.class_stats/sum(model.class_stats))
    print(model.pred_stats/sum(model.pred_stats))

if __name__ == "__main__":
    main()

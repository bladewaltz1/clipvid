import argparse
import logging
import os

import torch

from vid import VIDDataset
from vid_eval import do_vid_evaluation


def vid_evaluation(dataset, predictions, output_folder, motion_specific):
    logger = logging.getLogger("inference")
    logger.info("performing vid evaluation, ignored iou_types.")
    return do_vid_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        motion_specific=motion_specific,
        logger=logger,
    )


def inference_no_model(dataset, motion_specific=False, output_folder=None):
    predictions = torch.load(os.path.join(output_folder, "vid_val_30csize.pth"))
    print("prediction loaded.")
    return vid_evaluation(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    motion_specific=motion_specific)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--prediction-folder",
        help="The path to the prediction file to be evaluated.",
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
    )
    parser.add_argument(
        "--motion-specific",
        "-ms",
        action="store_true",
        help="if True, evaluate motion-specific iou"
    )
    args = parser.parse_args()

    dataset = VIDDataset(
        image_set = "VID_val_videos",
        data_dir=args.data_dir,
        anno_path=os.path.join(args.data_dir, "ILSVRC2015/Annotations/VID"),
        img_index=os.path.join(args.data_dir, "ILSVRC2015/ImageSets/VID_val_videos.txt")
    )
    inference_no_model(dataset, args.motion_specific, args.prediction_folder)


if __name__ == "__main__":
    main()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
from mobilenet_ssd_priors import priors
import torch.nn.functional as F
from utils import detect_objects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
n_classes = 20


def detect(model, original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = detect_objects(
        model,
        priors_cxcy,
        predicted_locs,
        predicted_scores,
        min_score=min_score,
        max_overlap=max_overlap,
        top_k=top_k,
        n_classes=n_classes,
    )

    # Move detections to the CPU
    det_boxes = det_boxes[0].to("cpu")

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [
            original_image.width,
            original_image.height,
            original_image.width,
            original_image.height,
        ]
    ).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to("cpu").tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ["background"]:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    # font = ImageFont.truetype("./calibril.ttf", 15)
    font = ImageFont.truetype("arial.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(
            xy=[l + 1.0 for l in box_location], outline=label_color_map[det_labels[i]]
        )  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2.0, box_location[1] - text_size[1]]
        textbox_location = [
            box_location[0],
            box_location[1] - text_size[1],
            box_location[0] + text_size[0] + 4.0,
            box_location[1],
        ]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill="white", font=font)
    del draw

    return annotated_image


def main(args):

    img_path = args.img_path

    # img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode="r")
    original_image = original_image.convert("RGB")
    # Load model checkpoint
    checkpoint = args.checkpoint
    checkpoint = torch.load(checkpoint, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(
        "\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n"
        % (start_epoch, best_loss)
    )
    model = checkpoint["model"]
    model = model.to(device)
    model.eval()

    detect(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="Image path")
    parser.add_argument("checkpoint", help="Path for pretrained model")
    args = parser.parse_args()

    main(args)

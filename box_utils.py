#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:00:00 2019

@author: viswanatha
"""

import collections
import torch
import itertools
from typing import List
import math

SSDBoxSizes = collections.namedtuple("SSDBoxSizes", ["min", "max"])

SSDSpec = collections.namedtuple(
    "SSDSpec", ["feature_map_size", "shrinkage", "box_sizes", "aspect_ratios"]
)


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([x_center, y_center, w, h])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([x_center, y_center, w, h])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([x_center, y_center, w * ratio, h / ratio])
                priors.append([x_center, y_center, w / ratio, h * ratio])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

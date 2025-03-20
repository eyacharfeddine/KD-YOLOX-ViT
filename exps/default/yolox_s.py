#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This config is adapted for license plate detection on a visible dataset.

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model scaling factors for YOLOX; adjust as necessary.
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- YOLOX with ViT Layer ---------------- #
        # Set to True to integrate a Vision Transformer layer for enhanced global feature extraction.
        self.vit = True

        # ---------------- Knowledge Distillation Config ---------------- #
        # Enable Knowledge Distillation to improve the performance of a smaller (student) model.
        self.KD = True
        # Use online KD: teacher inference is run on the fly during student training.
        self.KD_online = True
        # Folder to store teacher FPN logits (if using offline KD, otherwise optional).
        self.folder_KD_directory = "KD-FPN-Images/"

        # ---------------- Dataset Config ---------------- #
        # Set the dataset directory to where your visible license plate dataset is stored.
        # Make sure your dataset structure is as follows:
        #
        # datasets/Visible/Axis/
        #   ├── train2017/         # Contains training images (e.g., .jpg files)
        #   ├── val2017/           # Contains validation images
        #   └── annotations/       # Contains instances_train2017.json and instances_val2017.json
        #
        # Update these paths if needed.
        self.data_dir = "datasets/Visible/Axis/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        
        # For license plate detection, we assume one class.
        self.num_classes = 1

        # ---------------- Training & Evaluation Settings ---------------- #
        self.max_epoch = 400
        self.data_num_workers = 4
        self.eval_interval = 10

        # Optionally, adjust augmentation parameters:
        # self.mosaic_prob = 1.0
        # self.mixup_prob = 0.0
        # self.hsv_prob = 1.0
        # self.flip_prob = 0.5
        # self.degrees = 10.0
        # self.translate = 0.1
        # self.shear = 2.0


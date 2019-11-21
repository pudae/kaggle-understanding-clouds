#!/bin/bash

python tools/split.py
python tools/resize_images.py
python tools/resize_labels.py

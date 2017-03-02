#!/usr/bin/env bash

GPU_ID=0
DATA_DIR=/media/yves/sandisk3/caffe-master/examples/coco_caption/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

caffe train     -solver lstm_lm_solver.prototxt -gpu $GPU_ID

#!/usr/bin/env bash

GPU_ID=0
WEIGHTS=../cvgj/resnet50/resnet50_cvgj_iter_215764.caffemodel

#WEIGHTS=lrcn-resnet_iter_200000.caffemodel

#DATA_DIR=/media/yves/sandisk3/caffe-master/examples/coco_caption/h5_data/
#if [ ! -d $DATA_DIR ]; then
#    echo "Data directory not found: $DATA_DIR"
#    echo "First, download the COCO dataset (follow instructions in data/coco)"
#    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
#    exit 1
#fi

$CAFFE_HOME/build/tools/caffe train -solver lrcn-resnet_solver.prototxt -weights $WEIGHTS -gpu $GPU_ID

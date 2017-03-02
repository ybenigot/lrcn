#!/usr/bin/env bash

GPU_ID=0
WEIGHTS=lrcn-resnet_iter_200000.caffemodel

#caffe test 	-solver lrcn-resnet-test_solver.prototxt -weights $WEIGHTS 
#-gpu $GPU_ID
caffe test -model lrcn-resnet-test.prototxt -weights $WEIGHTS -gpu 0 -iterations 100
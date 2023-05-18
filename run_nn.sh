#!/bin/bash
echo "starting process..."
nohup /home/ahmadreza-n/miniconda3/envs/imitation/bin/python /home/ahmadreza-n/Documents/imitation-crp/nn.py data/data.csv data/events.csv output/model.onnx > output/output.log &
#!/bin/bash
echo "starting process..."
nohup /home/ahmadreza-n/miniconda3/envs/imitation/bin/python /home/ahmadreza-n/Documents/imitation-crp/main.py data/data.csv data/events.csv output/model.onnx gail 200 > output/output.log &
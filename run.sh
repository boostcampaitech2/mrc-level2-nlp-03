#!/bin/bash

while read line
do
    python train.py $line

done < train_command.txt

while read line
do
    python inference.py $line

done < inference_command.txt
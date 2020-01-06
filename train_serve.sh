#!/bin/sh

DIR=$(pwd)
# echo $OLDPWD
echo $DIR

docker pull chanwooda0223/marker1010-semantic-segmentation-serving


volume=$DIR
echo $volume

# docker run --gpus all -p 8893:8888 -p 8894:5000 -v volume/:/notebook -td train-serving /bin/bash

docker run --rm --gpus all -v $volume/:/notebook -t marker1010-semantic-segmentation-serving /bin/sh trainer.sh



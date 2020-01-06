#!/bin/sh

# DIR="$( cd "$( dirname "$0" )" && pwd -P )"
DIR=$(pwd)
# echo $OLDPWD
echo $DIR


service_version='cloud_v_90'
service_path='deploy/KerasSegmentationService/'$service_version'/'
service_abs_path=$DIR'/'$service_path

echo $service_path
echo $service_abs_path
service_name='marker1010-semantic-segmentation'

echo $service_abs_path

cd $service_abs_path
echo "dir move to "$service_abs_path


### docker building
docker build -t $service_name -f "./Dockerfile" .
echo "dockerization end"
echo $service_name

### docker run container
# docker run -p 
docker run -td -p 8892:5000 $service_name


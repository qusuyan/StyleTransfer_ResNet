#! /bin/bash

mkdir data
cd data
mkdir bin
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
cd ..

mkdir pretrained-model
cp ../imagenet-vgg-verydeep-19.mat pretrained-model/imagenet-vgg-verydeep-19.mat
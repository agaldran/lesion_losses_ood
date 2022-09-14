#!/usr/bin/env bash

wget https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip .
unzip hyper-kvasir-segmented-images.zip -d data_endotect/
rm hyper-kvasir-segmented-images.zip

python prepare_train_endo_data.py
rm -r data_endotect/segmented-images

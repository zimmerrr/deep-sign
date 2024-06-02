# DeepSign: A Deep Learning Mobile Application for Real-time Hand Gesture Recognition and Sign Language Interpretation	

## Introduction
A research project submitted to the faculty of Department of Computer Studies - Cavite State University Imus Campus

## Prerequisites
Before you begin, ensure you have met the following requirements:

Model Training/Customization
1. Python (3.10.14)
2. TensorFlow (2.16.1) 

## Developers
This project was developed by students of Cavite State University - Imus Campus as a part of fulfillment of the requirements for the subject of
COSC 200A - UNDERGRADUATE THESIS


1. Christian Andrei Torrijos - Development Lead
2. Jerome Joaquin - Developer
3. John Hendrix Macasiab - Developer


# Pre-requisites
install conda
conda create --name env_name python=3.12
conda activate env_name
conda install pytorch torchvision torchaudio cpuonly -c pytorch # Without GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip installtqdm datasets mediapipe wandb
pip install tqdm # for progress bar
pip install datasets # generate dataset from files

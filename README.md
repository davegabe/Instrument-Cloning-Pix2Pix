# Deep Audio Modelling
This repository contains the code for the project of the Big Data Computing course at the University of La Sapienza, Rome. For this course it was required to implement a Big Data application using the Apache Spark framework into a Google Colab notebook.

# Task description: Audio Modelling
For this project I decided to implement a deep learning application able to synthesize audio of an instrument given a sequence of annotated notes (audio + midi information). 
The application is based on a [Pix2Pix](https://arxiv.org/abs/1611.07004) architecture, which is a conditional GAN able to learn a mapping from input to output images. In this case the input is spectrogram of sawtooth/sinewave audio synthesized from midi information and the output is the spectrogram of the real audio of the instrument. 

# Dataset
The dataset used for this project is the [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth), which contains 305,979 musical notes from 1,006 instruments annotated with midi information such as pitch and velocity.

# Implementation
The application is implemented using the following libraries:
- [Apache Spark](https://spark.apache.org/): for the distributed computation of the training and testing of the model
- [Petastorm](https://petastorm.readthedocs.io/): for the distributed data loading
- [Pytorch Lightning](https://www.pytorchlightning.ai/): for the implementation of the training and testing loops
- [Pythorch Lightning Bolts](https://pytorch-lightning-bolts.readthedocs.io/en/latest/): for the implementation of the Pix2Pix model
- [Librosa](https://librosa.org/doc/latest/index.html): for the audio processing
- [Matplotlib](https://matplotlib.org/): for the plotting of the results
- [Frechet Audio Distance](https://github.com/gudgud96/frechet-audio-distance): for the computation of the Frechet Audio Distance (FAD) between the real and the generated audio
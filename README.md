# PyramidBox
This repo implements PyramidBox with pytorch
The paper [PyramidBox](https://arxiv.org/abs/1803.07737) is based on VGG16. But this repo is based on resnet50.

Here, I implements data-anchor-sampling, LFPN, max-in-out layer.

The result on WIDER FACE Val set:

AP Easy | AP Medium | AP Hard
--------|-----------|---------
  95.3  |    94.3   |  89.0   

## My model is released now!

You can download [model](https://pan.baidu.com/s/1tSys4yfvKEJVZcxTLzNbUw) from Baidu!

With the model and test.py, you can get the same result on WIDER FACE Val set!


## Usage
### Prerequisites

*Python3

*Pytorch3.1

*OpenCV3

*Anaconda: environment.yaml is the environment file, using ```conda env create -f environment.yaml``` to create the envrionment I test in my pc.

### For video / image
You can input the video or image

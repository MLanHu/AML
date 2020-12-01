## Introduction
This is a pytorch implementation of binary matching case described in *'Adversarial-Metric Learning for Audio-Visual Cross-Modal Matching'*  
![network](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/network.png)

## Requirements
python 3.6  
librosa 0.7.2  
numpy 1.19.0  
torch 1.4.0  
torchvision 0.5.0  

## Dataset
The network are trained on images from the and VGGFace dataset, and audio segments from VoxCeleb1. The VGGFace can be download from [here](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/zippedFaces.tar.gz). The VoxCeleb1 can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Experimental Result
### Comparison Results of the proposed method
Here are the comparison results of audio-visual matching against state-of-the-art methods on both binary (k = 2) and multi-way (k = 10) cases.
![Comparison Results](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/comparison_result.png)
### Qualitative Results of the proposed method
Here are the qualitative results of audio-visual cross-modal matching of the proposed AML comparing to DIMNet, SVHF-Net in A → V challenge with k = 2.
![Qualitative Results](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/qualitative_result.png)
### Video demonstration
Here is a video demo to demonstrate the results of the proposed method.
![video demonstration](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/Demo.mp4)
## Notice
The implementation of metric learning methods are included in
```
metric.py
```
The details of the  method can be found in 
```
@inproceedings{oh2016deep,
  title={Deep metric learning via lifted structured feature embedding},
  author={Oh Song, Hyun and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4004-4012},
  year={2016}
}
```


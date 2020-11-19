## Introduction
This is a pytorch implementation of binary matching case described in *'Adversarial-Metric Learning for Audio-Visual Cross-Modal Matching'*  
![image](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/network.png)

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
![image](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/comparison%20result.png)
### Qualitative Results of the proposed method
![image](https://github.com/MLanHu/AML/blob/main/Experimental%20Result/qualitative%20results.png)

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


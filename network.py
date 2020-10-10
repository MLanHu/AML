import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchvision import models

class G(nn.Module):
    def __init__(self,args):
        super(G, self).__init__()
        self.args = args
        self.audio = Audio_net()
        self.image = Frame_net()

        self.generator = nn.Sequential(
            nn.Dropout(self.args.dropout_g1),
            nn.Linear(self.args.feats_basic,self.args.hid_generator),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(self.args.dropout_g2),
            nn.Linear(self.args.hid_generator,self.args.feats_cls),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.args.dropout_c),
            nn.Linear(args.feats_cls*3, self.args.num_classes)
        )

    def forward(self,*input): # 0,1:audio, 2:image
        audio_feature1 = self.audio(input[0])
        audio_feature2 = self.audio(input[1])
        image_feature = self.image(input[2])
        audio_embedding1 = self.generator(audio_feature1)
        audio_embedding2 = self.generator(audio_feature2)
        image_embedding = self.generator(image_feature)
        out = self.classifier(torch.cat([image_embedding,audio_embedding1,audio_embedding2],1))
        return out,audio_embedding1,audio_embedding2,image_embedding

class D(nn.Module):
    def __init__(self,args):
        super(D, self).__init__()
        self.args = args

        self.discriminator = nn.Sequential(
            nn.Dropout(self.args.dropout_d1),
            nn.Linear(self.args.feats_cls,self.args.num_modality)
        )

    def forward(self,*input): # 0,1:audio, 2:image
        audio1_modality= self.discriminator(input[0])
        audio2_modality= self.discriminator(input[1])
        image_modality = self.discriminator(input[2])
        return audio1_modality,audio2_modality,image_modality

class Audio_net(nn.Module):
    def __init__(self):
        super(Audio_net,self).__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(1, 3, 5, (2,1), 1),
            nn.AvgPool2d(kernel_size=(3, 5), stride=2, ceil_mode=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(3, 5, (3,6), 2, 1),
            nn.BatchNorm2d(5),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(5,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(4, 5), stride=1, ceil_mode=False),
            nn.LeakyReLU(0.2, True),
             )
    def forward(self,a):
        out = self.transform(a)
        out = out.view(out.size(0),-1)
        return out

class Frame_net(nn.Module):
    def __init__(self):
        super(Frame_net, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(3,3,7,2,3),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(3,5,5,2,2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(5,8,2,1,4),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(8,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, True),
         )

    def forward(self,f):
        out = self.transform(f)
        out = out.view(out.size(0), -1)
        return out
from torch.utils.data import dataset
import librosa
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def get_txt(name):
    with open(name,'r') as f:
        data = [path for path in f]
    return data

class Dataset(dataset.Dataset):
    def __init__(self, dtype,args):
        self.transform = {
        'train':transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'test':transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'audio':transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
        }
        self.dtype = dtype
        if self.dtype == 'train':
            txt1 = args.trainfile
            data = get_txt(txt1)
            self.data = data
        else:
            txt = args.testfile
            self.data = get_txt(txt)


    def load_audio(self,audiopath):
        y, sr = librosa.load(audiopath)
        y = y - y.mean()
        y = self.preemphasis(y)
        y = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        y = y[:224,:125]
        y = self.transform['audio'](y)
        return y

    def preemphasis(self,signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])


    def load_image(self,PILimage):
        RealPILimage = Image.open(PILimage).convert("RGB")
        RealPILimage = self.transform[self.dtype](RealPILimage)
        return RealPILimage

    def __getitem__(self, index):
        image_path,audio1_path,audio2_path,label = self.data[index].split()
        image = self.load_image(image_path)
        audio1 = self.load_audio(audio1_path)
        audio2 = self.load_audio(audio2_path)
        label = int(label)
        image_m = 0
        audio_m = 1
        return image,audio1,audio2,label,audio_m,image_m

    def __len__(self):
        return len(self.data)


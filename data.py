from torch.utils.data import dataloader
from importlib import import_module
from torchvision import transforms
from dataset import Dataset




class Data:
    def __init__(self, args):
        self.dataset={ name:Dataset(name,args) for name in ['train','test'] }
        self.dataloader={}
        self.dataloader['train']=dataloader.DataLoader(self.dataset['train'],
                                                          shuffle=True,
                                                          batch_size=args.batchtrain,
                                                          num_workers=args.nThread,
                                                          pin_memory=args.pin_memory)
        self.dataloader['test']=dataloader.DataLoader(self.dataset['test'],
                                                          shuffle=False,
                                                          batch_size=args.batchtest,
                                                          num_workers=args.nThread,
                                                          pin_memory=args.pin_memory)


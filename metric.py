import torch
import torch.nn as nn

class lift_struct(nn.Module):
    def __init__(self,alpha,multi):
        super(lift_struct,self).__init__()
        self.alpha = alpha
        self.multi = multi # numbers of negative samples
    def forward(self, anchor, positive, neglist):
        batch = anchor.size(0)
        D_ij = torch.pairwise_distance(anchor, positive)
        D_in = 0 # distance between anchor and negative samples
        D_jn = 0 # distance between positive samples and negative samples
        for i in range(self.multi):
            a = torch.pairwise_distance(anchor,neglist[i])
            D_in += torch.exp(self.alpha - a)
            b = torch.pairwise_distance(positive,neglist[i])
            D_jn += torch.exp(self.alpha - b)
        D_n = D_in + D_jn
        J = torch.log(D_n) + D_ij
        J = torch.clamp(J,min=0)
        loss = J.sum()/(2*batch)
        return loss


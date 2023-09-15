import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import linecache #fast access to a specific file line
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import torch.nn.functional as F
import time
import torchinfo
from pathlib import Path

CONT_SIZE = 30

ALPHABET = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
ALPHABET = {ALPHABET[i]:i for i in range(len(ALPHABET))}
ALPHABET['-']= 20
ALPHABET['Z']= 21

class AttBlock(nn.Module):
    """
    Self-Attention Block of the Transformer
    """
    def __init__(self,in_features,out_features=None,num_heads=8,head_dims=24):
        super().__init__()
        out_features = out_features or in_features
        
        self.Q_w = nn.Linear(in_features,num_heads*head_dims,bias=False)
        self.K_w = nn.Linear(in_features,num_heads*head_dims,bias=False)
        self.V_w = nn.Linear(in_features,num_heads*head_dims,bias=False)
        
        self.att = nn.MultiheadAttention(num_heads*head_dims,num_heads=num_heads,batch_first=True)
        self.lin = nn.Linear(num_heads*head_dims,out_features)
        
    def forward(self,x):
        Q = self.Q_w(x)
        K = self.K_w(x)
        V = self.V_w(x)
        out,_ = self.att(Q,K,V,need_weights=False)
        out = self.lin(out)
        
        return out
    
class RowAttBlock(nn.Module):
    """
    Row-wise Attention Block
    """
    def __init__(self,in_features,out_features=None,num_heads=8,head_dims=24):
        super().__init__()
        self.att_block = AttBlock(in_features,num_heads=num_heads,head_dims=head_dims)
    
    def forward(self,x):
        b,n,l,d = x.shape
        out = self.att_block(x.view((b*n,l,d))).view(b,n,l,d)
        return out
    
class ColAttBlock(nn.Module):
    """
    Column-wise Attention Block
    """
    def __init__(self,in_features,out_features=None,num_heads=8,head_dims=24):
        super().__init__()
        self.att_block = AttBlock(in_features,num_heads=num_heads,head_dims=head_dims)
    
    def forward(self,x):
        b,n,l,d = x.shape
        out = self.att_block(x.view((b*l,n,d))).view((b,n,l,d)) #might not work
        return out
    
class FeedForward2D(nn.Module):
    """
    MLP Block of the Transformer
    """    
    def __init__(self,in_features,out_features=None,wide_factor=4):
        super().__init__()
        out_features = out_features or 2*in_features
        hidden_dim = wide_factor * 2*in_features
        
        self.lin1 = nn.Linear(2*in_features,hidden_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(hidden_dim,out_features)
        
    def forward(self,x):
        b,n,l,d = x.shape
        
        #dst = src.transpose(1, 2).reshape(b,l,n*d)
        #src = dst.reshape(b,l,-1,d).transpose(1,2)

        out = x.transpose(1, 2).reshape(b,l,n*d)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        out = out.reshape(b,l,-1,d).transpose(1,2)
        return out
    

class FeedForward(nn.Module):
    """
    MLP Block of the Transformer
    """    
    def __init__(self,in_features,out_features=None,wide_factor=4,act2 = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = wide_factor * in_features
        
        self.lin1 = nn.Linear(in_features,hidden_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(hidden_dim,out_features)
        
    def forward(self,x):
        out = self.lin1(x)
        out = self.act(out)
        out = self.lin2(out)
        
        return out
    
#(from the timm library)
def drop_path(x, drop_prob: float = 0.1, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Block(nn.Module):
    """
    Basic Block of the Transformer
    """
    def __init__(self,in_features,num_heads=8,head_dims=24,wide_factor=4,drop=0.1):
        super().__init__()
        
        self.row_att_block = AttBlock(in_features,num_heads=num_heads,head_dims=head_dims)
        self.ff2d = FeedForward(in_features,wide_factor=wide_factor)
        self.drop_path = DropPath(drop)
        
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        
    def forward(self,x):
        out = x + self.drop_path(self.row_att_block(x))
        out = self.norm1(out)
        out = out + self.drop_path(self.ff2d(out))
        out = self.norm2(out)
        
        return out
    

class Classifier_Head(nn.Module):
    """
    Classifier Head of the Transformer
    """
    def __init__(self,in_features,clf_dims,out_size,seq_len):
        super().__init__()
        in_dim = in_features
        self.in_dim = in_dim
        
        layers = []
        for out_dim in clf_dims:
            layers.append(nn.Linear(in_dim,out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=0.2))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim,out_size))
        
        self.clf = nn.Sequential(*layers)
        
    def forward(self,x):
        out = x.view((-1,self.in_dim))
        
        out = self.clf(out)
        
        return out
    

def get_params(input_size,N,head,head_dim,wide_factor,drop_prob):
    """
    Returns the initialization parameters of the Transformer
    """
    return input_size, [head for _ in range(N)], [head_dim for _ in range(N)], [wide_factor for _ in range(N)], [drop_prob for _ in range(N)], 

class AttNet(nn.Module):
    """
    Transformer-like neural net
    """
    def __init__(self,in_features,num_heads,head_dims,wide_factors,drops,embedding_dim=128,input_dim=21,out_size=20,num_seq=2,seq_len=2*CONT_SIZE+4,clf_dims=[256,64],cont_size=CONT_SIZE):
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim or in_features
        self.input_dim = input_dim
        self.cont_size=cont_size
        self.seq_len = seq_len
        
        blocks = []
        r = min(len(num_heads),len(head_dims),len(wide_factors),len(drops))
        for i,(n_h, h_d,w,d) in enumerate(zip(num_heads,head_dims,wide_factors,drops)):
            blocks.append(Block(embedding_dim,num_heads=n_h,head_dims=h_d,wide_factor=w,drop=d))

            
        self.feature_extractor = nn.Sequential(*blocks)
        self.clf = Classifier_Head(embedding_dim,clf_dims,out_size=20,seq_len=seq_len)
        
        sp = Path("../data/freq.pth")
        with sp.open("rb") as fp:
            self.F = nn.Parameter(torch.log(torch.load(fp)))
        
        pid_layers = [nn.Linear(1,in_features),nn.Sigmoid()]
        self.pid_l = nn.Sequential(*pid_layers)
        embed_layers =  [nn.Linear(in_features,embedding_dim),nn.GELU()]
        self.embedding = nn.Sequential(*embed_layers)
        self.el_average_weigths = nn.Parameter(torch.randn((1,seq_len,1)))
        self.seq_average_weigths = nn.Parameter(torch.randn((1,2,1,1)))

    def to_input(self,x,PID,pos,length,pfreqs,l_pfreqs):
        
        X_idx = torch.argmax(x[:,self.cont_size],dim=1)
        seq1 = x[:,:2*self.cont_size+1]
        y_freq = F.pad(F.softmax(self.F[X_idx],dim=1).unsqueeze(1), pad=(0, 1), mode='constant', value=0) 
        aa_pos = ((torch.arange(2*self.cont_size+1)-self.cont_size)/self.cont_size)
        aa_pos = repeat(aa_pos,"l -> b l",b=x.shape[0]).to(self.F.device)
        aa_pos = aa_pos.unsqueeze(2)
        pos_dim = 0 #(self.in_features-self.input_dim-1)//2
        
        for i in range(pos_dim): #positionnal_encoding
            p = torch.cos(pos[:,:2*self.cont_size+1]/(32**(2*i/pos_dim))).unsqueeze(2)
            ip = torch.sin(pos[:,:2*self.cont_size+1]/(32**(2*i/pos_dim))).unsqueeze(2)
            aa_pos = torch.cat([aa_pos,p,ip],dim=2)
        
        seq1 = torch.cat([seq1,aa_pos],dim=2)
        
        out = seq1
        
        pid = self.pid_l(PID.unsqueeze(1)).unsqueeze(1)
        #pid = rearrange(pid,"b 1 (n e) -> b n 1 e",n=2)
        
        pad = (0,self.in_features-self.input_dim+1)
        
        pfreqs = F.normalize(pfreqs,p=1,dim=1)
        pf = F.pad(pfreqs.unsqueeze(1),pad =pad, mode='constant', value=0)
        #pf = repeat(pf,"b 1 d -> b 2 1 d")
        l_pfreqs = F.normalize(l_pfreqs,p=1,dim=2)
        lpf = F.pad(l_pfreqs,pad=pad, mode='constant', value=0).unsqueeze(2)
        lpf = lpf[:,0]
        out = torch.cat([out,pf,pid,lpf],dim=1)
        
        return out
        
        
    def forward(self,x,PID,pos,length,pfreqs,l_pfreqs):
        X_input = self.to_input(x,PID,pos,length,pfreqs,l_pfreqs)
        X_input = self.embedding(X_input)
        features = self.feature_extractor(X_input)
        
        #weighted avergage + act function to break linearity
        
        w_el = F.softmax(self.el_average_weigths,dim=1) 
        clf_input = torch.atan((features * w_el).sum(dim=1))
        
        y_pred = self.clf(clf_input)
        return y_pred
    

class State:
    """
    Used for checkpointing
    """
    def __init__(self,model,optim,scheduler):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.epoch = 0
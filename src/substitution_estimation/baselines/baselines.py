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
import sys

from dataloading_utils_baselines import MyDataset, my_collate


CONT_SIZE = 30


ALPHABET = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

ALPHABET = {ALPHABET[i]:i for i in range(len(ALPHABET))}

ALPHABET['-']= 20
ALPHABET['Z']= 21

rep = torch.tensor([8, 8, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 7, 8, 8, 8, 8, 8])
rand = torch.tensor([0, 0, 0.2, 0, 0, 0, 0, 0, 0.2, 0.5, 0.3, 0.9, 0.8, 0.5, 0.9, 0, 0, 0, 0, 0])



fname = '../data/test_dataset_profile.pth'

savepath = Path(fname)
if not savepath.is_file():
    test_dataset = MyDataset(r"./data/test_data",cont_size = 6,div=2000)
    fname = './data/test_dataset_profile.pth'
    with savepath.open("wb") as fp:
        torch.save(test_dataset,fp)
else:
    with savepath.open("rb") as fp:
        test_dataset = torch.load(fp)


test_dataset.cont_size = CONT_SIZE
test_dataset.div = 2000



def PID_Freqs_pred(X,PID=None,LPID=None,freq=None,cont_size=CONT_SIZE):
    savepath = Path("./data/freq.pth")
    with savepath.open("rb") as fp:
        FREQS = torch.load(fp)

    PID = rearrange(PID, "b -> b 1")
    y_hat = X[:,cont_size][:,:-1]*PID
    X_idx = torch.argmax(X[:,cont_size][:,:-1],dim=1)

    mask = torch.ones_like(y_hat) - X[:,cont_size][:,:-1]
    norm_coef = FREQS[X_idx] * X[:,cont_size][:,:-1]
    y_2 = mask * FREQS[X_idx]

    norm_coef = (1-torch.sum(norm_coef,dim=1))
    norm_coef = rearrange(norm_coef, "b -> b 1")


    y_2 = y_2 / norm_coef
    y_2 = y_2 * (1-PID)

    return y_hat + y_2

def profile_pred(profiles):
    inp = (profiles[:,CONT_SIZE]-1).clip(min=0)
    y_hat = F.normalize(inp,dim=1,p=1)
    return y_hat

def PID_profile_pred(X,PID,profile,cont_size=CONT_SIZE):
    inp = (profile[:,CONT_SIZE]-1).clip(min=0)
    profile_freq = F.normalize(inp,dim=1,p=1)

    PID = rearrange(PID, "b -> b 1")
    y_hat = X[:,cont_size][:,:-1]*PID
    X_idx = torch.argmax(X[:,cont_size][:,:-1],dim=1)

    mask = torch.ones_like(y_hat) - X[:,cont_size][:,:-1]
    norm_coef = profile_freq[X_idx] * X[:,cont_size][:,:-1]
    y_2 = mask * profile_freq[X_idx]

    norm_coef = (1-torch.sum(norm_coef,dim=1))
    norm_coef = rearrange(norm_coef, "b -> b 1")


    y_2 = y_2 / norm_coef
    y_2 = y_2 * (1-PID)

    return y_hat + y_2



R_str = """0.425093
0.276818 0.751878
0.395144 0.123954 5.076149
2.489084 0.534551 0.528768 0.062556
0.969894 2.807908 1.695752 0.523386 0.084808
1.038545 0.363970 0.541712 5.243870 0.003499 4.128591
2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847
0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484
0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882
0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067
0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500
1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604
0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853
1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464
4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132
2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279
0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825
0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815
2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313
"""

PI_str = """0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147"""

PI = torch.tensor(list(map(float,PI_str.split(' '))))

R = torch.zeros(20,20)
i = 0
for line in R_str.splitlines():
    i += 1
    l = list(map(float,line.split(' ')))
    for j in range(i):
        R[i,j] += l[j]
        R[j,i] += l[j]

mu = 0
Q = torch.zeros(20,20)
for i in range(20):
    for j in range(i+1,20):
        Q[i,j] = PI[i]*R[i,j]
        Q[j,i] = PI[j]*R[j,i]
        
    Q[i,i] = -torch.sum(Q[i])
    mu -= PI[i] * Q[i,i]

LG_Q_norm = Q/mu



def TR_mat_clf(X,ids,couples,freqs,dataset,cont_size=CONT_SIZE,Q_norm=LG_Q_norm):
    """
    Need to precompute all pairwise distances using FastME.
    """
    dist_data_dir = "./data/test_data_dist"

    X_idx = torch.argmax(X[:,cont_size],dim=1) #OneHot -> {0,...,19} #X and y are aligned bcs X \in {0,...,19}
    y_hat = []

    for ex in range(len(ids)):
        i,j = couples[ex] 
        #fetch distance t
        data_path = os.path.join(dist_data_dir, dataset.paths[ids[ex].item()][:-5]+"dist") #fetching file containing dist matrix
        line_i = linecache.getline(data_path, 2+i)[:-1] #starting at 2 bcs linecache start counting at 1 and first line is seq_num
        dists = line_i.split('   ')
        assert len(dists) == 1+int(linecache.getline(data_path, 1))
        
        try:
            t = float(dists[1+j]) # fetching the distance 
        except:
            print(i,j,ids[ex].item())
            return 0
        
        #compute Q*t
        mat = Q_norm*t 
        #compute e^Qt
        probs = torch.matrix_exp(mat)
        #access X
        pred = probs[X_idx[ex]] #access the correct row
        #assert (pred>0).all()
        #assert torch.isclose(torch.sum(pred),torch.tensor(1.))
        y_hat.append(pred)
    linecache.clearcache()
    return torch.stack(y_hat)
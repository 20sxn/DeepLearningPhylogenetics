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

from models_CrossCorr_utils import *

torch.multiprocessing.set_sharing_strategy('file_system') #to avoid issues in the dataloading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONT_SIZE = 30

ALPHABET = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

ALPHABET = {ALPHABET[i]:i for i in range(len(ALPHABET))}

ALPHABET['-']= 20
ALPHABET['X']= 21


def printalifile(s1,s2,sa,seqname1,seqname2,log_fname):
    with open(log_fname, 'a+') as f:
        print(seqname1,file=f)
        ii = 0
        seq1 = ''
        for i in range(len(sa)):
            if sa[i] == 'd' or sa[i] == 'g':
                seq1 += s1[ii]
                ii = ii + 1
            else:
                seq1 +='.'
        print(seq1,file=f)
        print(seqname2,file=f)
        ii = 0
        seq2 = ''
        for i in range(len(sa)):
            if sa[i] == 'd' or sa[i] == 'h':
                seq2 += s2[ii]
                ii = ii + 1
            else:
                seq2 +='.'
        print(seq2,file=f)



fname = "../substitution_estimation/models/state_1seq_CrossCorrCONV.pth"
savepath = Path(fname)
if savepath.is_file():
    with savepath.open("rb") as fp:
        model = torch.load(fp).model.eval().to(device)


def score_model(s1,s2,k1,k2,storage,dist='muscle',model=model,pfreqs=torch.tensor([0.0853,0.0559,0.0390,0.0556,0.0136,0.0370,0.0642,0.0704,0.0220,0.0601,0.1015,0.0531,0.0218,0.0423,0.0442,0.0629,0.0538,0.0138,0.0327,0.0708])):
    
    #model(x,kmer_sim,pfreqs,l_pfreqs)
    #[(b,2*CONT_SIZE+1,21),(b,),(b,20),(b,20)]
    if len(storage) < len(s1):
        
        lpfreq = torch.ones((2,20),device=device)
        pfreqs = pfreqs.to(device).unsqueeze(0)
        
        for aa1 in s1:
            idx = ALPHABET.get(aa1, 21)
            if idx < 21:
                lpfreq[0][idx]+=1
                
        for aa1 in s2:
            idx = ALPHABET.get(aa1, 21)
            if idx < 21:
                lpfreq[1][idx]+=1
                
        unknown1 = ''
        unknown2 = ''
        for aa in s1:
            if aa == 'B':
                unknown1 += 'N'
                unknown2 += 'D'

            elif aa == 'Z':
                unknown1 += 'Q'
                unknown2 += 'E'
            else:
                unknown1 += 'X'
                unknown2 += 'X'
        
                
        full_seq1 = [ALPHABET.get(i, 21) for i in s1]
        unk1 = [ALPHABET.get(i, 21) for i in unknown1]
        unk2 = [ALPHABET.get(i, 21) for i in unknown2]
        
        full_seq1 = torch.tensor(full_seq1)
        unk1 = torch.tensor(unk1)
        unk2 = torch.tensor(unk2)
        
        full_seq1 = F.one_hot(full_seq1,22).float()
        unk1 = F.one_hot(unk1,22).float()
        unk2 = F.one_hot(unk2,22).float()
        
        full_seq1 += (unk1+unk2)/2
        full_seq1[:,21] = 0.
        full_seq1= full_seq1.to(device)
        
        unknown1 = ''
        unknown2 = ''
        for aa in s2:
            if aa == 'B':
                unknown1 += 'N'
                unknown2 += 'D'

            elif aa == 'Z':
                unknown1 += 'Q'
                unknown2 += 'E'
            else:
                unknown1 += 'X'
                unknown2 += 'X'
        
        
        full_seq2 = [ALPHABET.get(i, 21) for i in s2]
        unk1 = [ALPHABET.get(i, 21) for i in unknown1]
        unk2 = [ALPHABET.get(i, 21) for i in unknown2]
        
        full_seq2 = torch.tensor(full_seq2)
        unk1 = torch.tensor(unk1)
        unk2 = torch.tensor(unk2)
        
        full_seq2 = F.one_hot(full_seq2,22).float()
        unk1 = F.one_hot(unk1,22).float()
        unk2 = F.one_hot(unk2,22).float()
        
        full_seq2 += (unk1+unk2)/2
        full_seq2[:,21] = 0.
        full_seq2 = full_seq2.to(device)
        
        X = []
        X1 = []
        X2 = []
        for k in range(len(s1)):
            window_i = ''
            unknown1 = ''
            unknown2 = ''
            for w in range(k-CONT_SIZE,k+CONT_SIZE+1):
                if w < 0 or w >= len(s1): #case of the edges
                    window_i += 'X'
                else:
                    window_i += s1[w]
                    
                if window_i[-1] == 'B':
                    unknown1 += 'N'
                    unknown2 += 'D'
                
                elif window_i[-1] == 'Z':
                    unknown1 += 'Q'
                    unknown2 += 'E'
                else:
                    unknown1 += 'X'
                    unknown2 += 'X'
                
            X_k = [ALPHABET.get(i, 21) for i in window_i]       
            X.append(X_k)
            X1.append([ALPHABET.get(i, 21) for i in unknown1])
            X2.append([ALPHABET.get(i, 21) for i in unknown2])
            
        X = torch.tensor(X)
        X1 = torch.tensor(X1)
        X2 = torch.tensor(X2)
        
        X = F.one_hot(X,22)[:,:,0:-1].float()
        X1 = F.one_hot(X1,22)[:,:,0:-1].float()
        X2 = F.one_hot(X2,22)[:,:,0:-1].float()
        
        X += (X1+X2)/2
        
        X = X.to(device)
            
        len_s1 = len(X)

        full_seq1 = full_seq1.unsqueeze(0)
        full_seq2 = full_seq2.unsqueeze(0)
        lpfreq = lpfreq.repeat((len_s1,1,1))
        pfreqs = pfreqs.repeat((len_s1,1))
        
        
        storage = F.softmax(model.forward_alignment(X,full_seq1,full_seq2,pfreqs,lpfreq),dim=1)
        storage = storage.cpu().detach()
        
        
    
    pfreqs=torch.tensor([0.0853,0.0559,0.0390,0.0556,0.0136,0.0370,0.0642,0.0704,0.0220,0.0601,0.1015,0.0531,0.0218,0.0423,0.0442,0.0629,0.0538,0.0138,0.0327,0.0708])
    
    if s2[k2] == "X":
        score = [0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1][{'A': 0,'R': 1,'N': 2,'D': 3,'C': 4,'Q': 5,'E': 6,'G': 7,'H': 8,'I': 9,'L': 10,'K': 11,'M': 12,'F': 13,'P': 14,'S': 15,'T': 16,'W': 17,'Y': 18,'V': 19,'B': 20,'Z': 21,'X': 22}[s1[k1]]]
        score = torch.tensor(score)
    
    elif s2[k2] in ALPHABET:
        obs = torch.log(storage[k1][ALPHABET[s2[k2]]])
        expec =  torch.log(pfreqs[ALPHABET[s2[k2]]]) #torch.log(pfreqs[ALPHABET[s1[k1]]]) +
        score = obs - expec
    
    elif s2[k2] == "B":
        obs = torch.log((storage[k1][ALPHABET['N']]+storage[k1][ALPHABET['D']])/2)
        expec =  torch.log((pfreqs[ALPHABET['N']]+pfreqs[ALPHABET['D']])/2)
        score = obs - expec
        
    elif s2[k2] == "Z":
        obs = torch.log((storage[k1][ALPHABET['Q']]+storage[k1][ALPHABET['E']])/2)
        expec =  torch.log((pfreqs[ALPHABET['Q']]+pfreqs[ALPHABET['E']])/2)
        score = obs - expec
        
    
    
    else:
        print(s2[k2])
        
        
    return score.item(),storage

    
def alignit_Elast(s1,s2,gOp=-10, gExt=-1,subs={},scorefunc=score_model,lmda=3.3):
    # matrice des distances
    m = list(range(len(s2)+1))
    # matrice des chemins
#    c = range(len(s1)+1)
    for i in range(len(s2)+1):
        m[i] = list(range(len(s1)+1))
        #c[i] = range(len(s2)+1)
    # root cell
    m[0][0] = (0, 'o')
    m[0][1] = (0, 'g', 1)
    m[1][0] = (0, 'h', 1)
    
    # first line #first gap is free
    for j in range(2,len(s1)+1):
        m[0][j]=(0, 'g', 1)
#        c[0][j]= 'h'  # insertion ds s1
  
    # first column
    for i in range(2,len(s2)+1):
        m[i][0]=(0 , 'h', 1)
#        c[i][0]= 'd' # deletion ds s1
    storage = []
    #max en colonne
    maxCol=[0]*(len(s1)+1)
    # tab
    for i in range(1,len(s2)+1):
        maxLigne=0
        for j in range(1,len(s1)+1):
            #substitution
            #distd = scorefunc(s2[i-1], s1[j-1], )
            dist,storage = scorefunc(s1,s2,j-1,i-1,storage)
            dist = lmda * dist
            distd = dist + m[i-1][j-1][0]
            
            #Calcul des gaps selon g=gOp+(l-1)*gExt
            #Si on a un gap avant:
            gap=gOp+gExt*(i-maxCol[j]-1)
            disth = gap + m[maxCol[j]][j][0]
            gap=gOp+gExt*(j-maxLigne-1)
            distg = gap + m[i][maxLigne][0]
            
            #end gaps
            if j == len(s1):
                disth = m[i-1][j][0]
            if i == len(s2):
                distg = m[i][j-1][0]
                
            if distd >= disth and distd >= distg:
                m[i][j] = (distd,'d')
#                c[i][j] = 'd'  # substitution
            elif disth >= distd and disth >= distg:
                m[i][j] = (disth, 'h', i-maxCol[j])
#                c[i][j] = 'h' # indel
            else:
                m[i][j] = (distg, 'g', j-maxLigne)
#                c[i][j] = 'g' # indel
            if m[i][j][0] > m[maxCol[j]][j][0]:
                maxCol[j]=i
            if m[i][j][0] > m[i][maxLigne][0]:
                maxLigne=j
    return m

def backtrack(m):
    #print "distance = ",m[len(s1)][len(s2)]
    #print m
    #print c
    # backtrack
    #nb col
    j = len(m[0])-1
    #nb lignes
    i = len(m)-1
    chemin = ''
    while (i != 0 or j != 0):
        if i < 0 or j < 0:
            print("backtrack:: ERROR i or j <0",i,j)
            exit(1)
        #print i,j,m[i][j]
        if m[i][j][1] == 'd':
            i = i-1
            j = j-1
            chemin = 'd'+chemin
        elif m[i][j][1] == 'g':
            if len (m[i][j]) ==3:
                #print "Saut de ",m[i][j][2]
                for k in range(m[i][j][2]):
                    j=j-1 
                    chemin = 'g'+chemin
            else:
                j = j-1
                chemin = 'g'+chemin
        else:
            if len (m[i][j]) ==3:
                #print "Saut de ",m[i][j][2]
                for k in range(m[i][j][2]):
                    i = i-1
                    chemin = 'h'+chemin
            else:
                i = i-1
                chemin = 'h'+chemin
        #print i,j
    #print chemin
    return chemin



def fd_score(s1_test,s2_test,s1_ref,s2_ref,gap_test='-',gap_ref='.'):
    align_len = 0
    test1 = []
    test2 = []
    cpt1 = 1
    cpt2 = 1
    for i1,i2 in zip(s1_test,s2_test):
        and_cpt = 0
        if i1 != gap_test:
            test1.append(cpt1)
            cpt1 += 1
        else:
            test1.append(0)
        if i2 != gap_test:
            and_cpt+= 1
            test2.append(cpt2)
            cpt2 += 1
        else:
            test2.append(0)
        

            
    ref1 = []
    ref2 = []
    cpt1 = 1
    cpt2 = 1
    for i1,i2 in zip(s1_ref,s2_ref):
        and_cpt = 0
        if i1 != gap_ref:
            if i1.isupper():
                ref1.append(cpt1)
                and_cpt += 1
            else:
                ref1.append(0)
            cpt1 += 1
        else:
            ref1.append(0)
        if i2 != gap_ref:
            if i2.isupper():
                ref2.append(cpt2)
                and_cpt += 1
            else:
                ref2.append(0)
            cpt2 += 1
        else:
            ref2.append(0)
        
        if and_cpt == 2:
            align_len += 1

    test = {(test1[i],test2[i]) for i in range(len(test1)) if (test1[i] and test2[i])}
    ref = {(ref1[i],ref2[i]) for i in range(len(ref1)) if (ref1[i] and ref2[i])}

    return (len(test.intersection(ref))/align_len)
        


def main(in_dir_path,out_dir_path):
    for path in os.listdir(in_dir_path):
        # check if current path is a file
        temp_path = os.path.join(in_dir_path, path)
        if os.path.isfile(temp_path):
            log_fname = os.path.join(out_dir_path, path)
            with open(temp_path, newline='') as f:
                s1 = ''
                s2 = ''
                line = f.readline()[:-1]
                cpt = 0
                seqname1 = ''
                seqname2 = ''
                while line:
                    if line[0] == '>':
                        if cpt == 0:
                            seqname1 = line
                        else:
                            seqname2 = line
                        cpt += 1
                    else:
                        if cpt < 2:
                            s1 += line
                        else:
                            s2 += line

                    line = f.readline()[:-1]
                
            mat = alignit_Elast(s1,s2,gOp=-10, gExt=-1)
            sa = backtrack(mat)
            printalifile(s1,s2,sa,seqname1,seqname2,log_fname)

if __name__ == "__main__":
    arguments = sys.argv
    assert len(arguments) > 1
    
    in_dir_path = arguments[1]

    if len(arguments) > 2:
        out_dir_path = arguments[2]
    else:
        out_dir_path = "out"

    main(in_dir_path,out_dir_path)
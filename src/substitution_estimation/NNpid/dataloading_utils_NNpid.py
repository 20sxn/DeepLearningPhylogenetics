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

torch.multiprocessing.set_sharing_strategy('file_system') #to avoid issues in the dataloading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CONT_SIZE = 30


ALPHABET = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

ALPHABET = {ALPHABET[i]:i for i in range(len(ALPHABET))}

ALPHABET['-']= 20
ALPHABET['Z']= 21

rep = torch.tensor([8, 8, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 5, 7, 8, 8, 8, 8, 8])
rand = torch.tensor([0, 0, 0.2, 0, 0, 0, 0, 0, 0.2, 0.5, 0.3, 0.9, 0.8, 0.5, 0.9, 0, 0, 0, 0, 0])

class MyDataset(Dataset):
    def __init__(self, data_dir, cont_size=6,div=2000,verbose=False):
        """
        Initialize the dataset by precomputing a bunch of data on the sequence families
        """
        self.col_size = 60 #number of column per file (Fasta standard)
        self.data_dir = data_dir #directory of the dataset
        self.cont_size = cont_size
        self.div = div
        self.len = 0  #number of families of sequences (1 per file)
        self.paths = [] #path of each families in the folder
        self.seq_lens = [] #length of each member of the family
        self.seq_nums = [] #number of member of the family
        self.aa_freqs = [] #frequencies of each symbol in the sequence family
        self.p_aa_freqs = [] #frequencies of each symbol in each sequence of a family
        
        
        dir_path = data_dir
        count = 0
        
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            temp_path = os.path.join(dir_path, path)
            if os.path.isfile(temp_path):
                n = 0 #number of sequences
                p = 0 # used to calculate the length of the sequences
                r = 0 # also used this way

                l = 0 # length of the seq l = p * self.col_size + r 

                cpt = 0 # to detect inconsistencies
                
                with open(temp_path, newline='') as f:
                    first_prot = True
                    newf = True
                    
                    aa_freq = torch.ones(20)
                    p_aa_freq = torch.ones(0)
                    
                    #parsing the file
                    line = f.readline()[:-1]
                    while line:
                        cpt += 1
                        if line[0] == '>': #header line
                            if not first_prot:
                                p_aa_freq = torch.cat([p_aa_freq,prot_aa_freq])
                            prot_aa_freq = torch.zeros(1,20)
                            n += 1
                            if newf and not first_prot:
                                newf = False
                            first_prot = False
                                
                        else:# sequence line
                            if newf and len(line) == self.col_size:
                                p += 1

                            if newf and len(line) != self.col_size:
                                r = len(line)
                            for aa in line:
                                aa_id = ALPHABET.get(aa,21)
                                if aa_id < 20:
                                    aa_freq[aa_id] += 1
                                    prot_aa_freq[0][aa_id] += 1

                            assert len(line) == self.col_size or len(line) == r
                        line = f.readline()[:-1]
                    
                    p_aa_freq = torch.cat([p_aa_freq,prot_aa_freq])
                    #aa_freq = F.normalize(aa_freq,dim=0,p=1)
                    #p_aa_freq = F.normalize(p_aa_freq,dim=1,p=1)

                l = p*self.col_size + r
                
                #sanity check
                #if the file line count is coherent with the number of sequences and their line count
                try: #if r != 0
                    assert (p+2) * n == cpt
                except: #if r == 0
                    assert (p+1) * n == cpt
                    assert r == 0
                    
                
                if n>1: #if this is false, we can't find pairs
                    self.paths.append(path)
                    self.seq_lens.append(l)
                    self.seq_nums.append(n)
                    self.aa_freqs.append(aa_freq)
                    self.p_aa_freqs.append(p_aa_freq)
                    count += 1
                    
                    if verbose and (count % 100 ==0) : print(f"seen = {count}")
            
        self.len = count
    
    def __len__(self):
        return self.len
     
    def sample(self, high, low=0, s=1):
        sample = np.random.choice(high-low, s, replace=False)
        return sample + low
    
    def __getitem__(self, idx, sample_size='auto',rep=rep,rand=rand): 
        """
        input idx of the family of the sample
        return a Tensor containing several samples from the family corresponding to the index
        """
        
        X = []
        y = []
        
        PIDs = []
        local_PIDs = []
        
        lengths = []
        
        pos = []
        
        pfreqs = []
        local_pfreqs = []
        
        bin_n = len(rep) #for biasing the sampling
        
        precomputed_pos = [] #positions of the amino-acids
        for i in range(-self.cont_size,self.cont_size+1):
            precomputed_pos.append(i)
        for i in range(-self.cont_size,0):
            precomputed_pos.append(i)
        for i in range(1,self.cont_size+1):
            precomputed_pos.append(i)
        
        precomputed_pos = torch.tensor(precomputed_pos).float()
        
        data_path = os.path.join(self.data_dir, self.paths[idx])
        try:
            n = self.seq_nums[idx]
            l = self.seq_lens[idx]
        except:
            print(idx)
            pass
        
        #sampling more for big families and long sequences
        if type(sample_size) != int:
            sample_s = min(n * l,25_000)
            coef = round((sample_s)/self.div) 
            sample_size = max(1,coef)
        
        p = l // self.col_size
        r = l % self.col_size # l = p * q + r
        sequence_line_count = p+2 if r else p+1

        for _ in range(sample_size):
            i,j = self.sample(n,s=2)

            start_i = 2 + (sequence_line_count)*i #start line of protein i
            start_j = 2 + (sequence_line_count)*j #start line of protein j
            
            seq_i = ''
            seq_j = ''
            
            PID_ij = 0
            
            l_ij = 0
            for offset in range(sequence_line_count-1): #computing PID and removing aligned '-' ##might need to compute the actual column num
                line_i = linecache.getline(data_path, (start_i + offset))[:-1]
                line_j = linecache.getline(data_path, (start_j + offset))[:-1]
                for aa_i, aa_j in zip(line_i,line_j):
                    if aa_i == aa_j:
                        if aa_i != '-':
                            PID_ij += 1
                            seq_i += aa_i
                            seq_j += aa_j        
                    else:
                        seq_i += aa_i
                        seq_j += aa_j
                    
                    if aa_j != '-' and aa_i != '-':
                        l_ij += 1
            
            try:
                PID_ij = PID_ij/l_ij
            except:
                PID_ij = 0 #case 0/0
            
            align_l = len(seq_i)
            possible_k = [] #possible position to take
            for k,(a_i,a_j) in enumerate(zip(seq_i,seq_j)):   
                if ALPHABET.get(a_i,21) < 20 and ALPHABET.get(a_j,21) < 20:
                    possible_k.append(k)
            
            # biasing for more diverse PID  
            bin_idx = int(PID_ij//(1/bin_n))
            rep_number = rep[bin_idx].clone()
            if torch.rand(1) < rand[bin_idx]:
                rep_number+=1
            
            for _ in range(rep_number):
                try:   
                    k = np.random.choice(possible_k)
                except:
                    continue
                
                #adding to the output
                lengths.append(align_l)
                pos_ij = (k + precomputed_pos)
                pos.append(pos_ij)
                
                #computing the windows
                window_i = ''
                window_j = ''
                for w in range(k-self.cont_size,k+self.cont_size+1):
                    if w < 0 or w >= align_l: #case of the edges
                        window_i += 'Z'
                        window_j += 'Z'
                    else:
                        window_i += seq_i[w]
                        window_j += seq_j[w]

                y_j = ALPHABET.get(window_j[self.cont_size], 21) # 'Z' is the default value for rare AA
                X_i = [ALPHABET.get(i, 21) for i in (window_i+window_j[:self.cont_size]+window_j[self.cont_size+1:])]       

                X.append(X_i)
                y.append(y_j)
                PIDs.append(PID_ij)
                #computing the local PID
                local_PID_ij = sum(1 for AA1,AA2 in zip(window_i, window_j[:self.cont_size]) if AA1 == AA2 and ALPHABET.get(AA1,21) < 20) \
                             + sum(1 for AA1,AA2 in zip(reversed(window_i), reversed(window_j[self.cont_size+1:])) if AA1 == AA2 and ALPHABET.get(AA1,21) < 20)

                loc_comp = sum(1 for AA1,AA2 in zip(window_i, window_j[:self.cont_size]) if ALPHABET.get(AA1,21) < 20 and ALPHABET.get(AA2,21) < 20) \
                             + sum(1 for AA1,AA2 in zip(reversed(window_i), reversed(window_j[self.cont_size+1:])) if ALPHABET.get(AA1,21) < 20 and ALPHABET.get(AA2,21) < 20)
                try:
                    tmp = local_PID_ij/loc_comp  
                except:
                    tmp = 0 #case 0/0

                local_PIDs.append(tmp)
                
                fam_freqs = self.aa_freqs[idx].clone()
                fam_freqs[y_j] -= 1
                pfreqs.append(fam_freqs)
                
                p_i_freqs = self.p_aa_freqs[idx][i].clone()
                p_j_freqs = self.p_aa_freqs[idx][j].clone()
                p_j_freqs[y_j] -= 1
                
                local_pfreqs.append(torch.stack((p_i_freqs,p_j_freqs)))

                assert y_j < 20
                assert X_i[self.cont_size] < 20
            
        linecache.clearcache() #clearing the cache
        X = torch.tensor(X)
        try:
            X = F.one_hot(X,22)[:,:,0:-1]
        except RuntimeError:
            pass
        if len(pos) == 0:
            pos = torch.tensor(pos)
        else:
            pos = torch.stack(pos)

        X = X.float()
        y = torch.tensor(y)
        PIDs = torch.tensor(PIDs)
        local_PIDs = torch.tensor(local_PIDs)
        lengths = torch.tensor(lengths)
        pfreqs = torch.stack(pfreqs)
        local_pfreqs = torch.stack(local_pfreqs)
        
        out = X,y.long(),PIDs,local_PIDs,pos,lengths,pfreqs,local_pfreqs
        return out




def my_collate(batch):
    """
    Transforms a list of tensors to a batch tensor
    """
    data = torch.cat([item[0] for item in batch],dim=0)
    target = torch.cat([item[1] for item in batch],dim=0)
    PID = torch.cat([item[2] for item in batch],dim=0)
    lPID = torch.cat([item[3] for item in batch],dim=0)
    pos = torch.cat([item[4] for item in batch],dim=0)
    length = torch.cat([item[5] for item in batch],dim=0)
    pfreqs = torch.cat([item[6] for item in batch],dim=0)
    l_pfreqs = torch.cat([item[7] for item in batch],dim=0)
    return data, target, PID, lPID, pos, length, pfreqs, l_pfreqs
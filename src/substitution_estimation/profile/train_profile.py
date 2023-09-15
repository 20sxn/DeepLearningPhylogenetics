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


from dataloading_utils_profile import MyDataset, my_collate

from substitution_estimation.profile.models_profile_utils import AttNet, State, get_params

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

def training_loop(train_loader,val_loader,epochs=101,fname="../models/state.pth",fnameb=None,state=None,last_epoch_sched=float('inf'),use_mut=True,cont_size=CONT_SIZE):
    """
    Trains a model
    """
    #to get the best model
    best = float('inf')
    
    #getting the acceleration device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #loading from previous checkpoint
    if fnameb is None:
        fnameb = fname[:-4] + '_best' +fname[-4:]
        
    savepath = Path(fname)
    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
    
    assert state is not None

    log_fname = fname[:-4]+'_log.txt'
    
    Loss = nn.CrossEntropyLoss(reduction='sum')
    
    #for logs
    List_Loss = []
    Eval_Loss = []
    for epoch in range(state.epoch, epochs):
        batch_losses = []
        state.model.train()
        for X,y,PID,lPID,pos,length,pfreqs,l_pfreqs,profiles in train_loader:
            X = X.to(device)
            y = y.to(device)
            PID = PID.to(device)
            profiles = profiles.to(device)
            
            state.optim.zero_grad()
            y_hat = state.model(X,PID,profiles)
            l = Loss(y_hat,y)/440 
            l.backward()
            state.optim.step()
            
            
            
            batch_losses.append(l.detach().cpu())
        List_Loss.append(torch.mean(torch.stack(batch_losses)).detach().cpu())
        state.epoch = epoch + 1
        if epoch < last_epoch_sched:
            state.scheduler.step()
        
        savepath = Path(fname)
        with savepath.open("wb") as fp:
            torch.save(state,fp)
        
        with torch.no_grad():
            eval_losses = [] 
            state.model.eval()
            for X,y,PID,lPID,pos,length,pfreqs,l_pfreqs,profiles in val_loader:

                X = X.to(device)
                y = y.to(device)
                PID = PID.to(device)
                profiles = profiles.to(device)

                y_hat = state.model(X,PID,profiles)
                y_hat = F.softmax(y_hat,dim=1)
                
                y = F.one_hot(y, 20)
                eval_l = (y_hat-y)**2
                eval_losses.append(torch.sum(eval_l,dim=1).detach().cpu())
    
            score = torch.mean(torch.cat(eval_losses)).detach().cpu().item()
            Eval_Loss.append(score)
        
        if score < best :
            best = score
            savepath = Path(fnameb)
            with savepath.open("wb") as fp:
                torch.save(state,fp)
                
        if epoch in {495,1007,2031,4079}:
            savepath = Path(fname[:-4]+'_'+str(epoch)+'.pth')
            with savepath.open("wb") as fp:
                torch.save(state,fp)
        
        with open(log_fname, 'a+') as f:
            print(f"epoch n°{epoch} : train_loss = {List_Loss[-1]}, val_loss = {Eval_Loss[-1]}",file=f)
                 
    return List_Loss,Eval_Loss,state


if __name__ == "__main__":



    fname = '../data/test_dataset_profile.pth'
    savepath = Path(fname)
    if not savepath.is_file():
        test_dataset = MyDataset(r"../data/test_data",cont_size = CONT_SIZE,div=2000)
        with savepath.open("wb") as fp:
            torch.save(test_dataset,fp)
    else:
        with savepath.open("rb") as fp:
            test_dataset = torch.load(fp)

    fname = '../data/train_dataset_profile.pth'
    savepath = Path(fname)
    if not savepath.is_file():
        train_dataset = MyDataset(r"../data/train_data",cont_size = CONT_SIZE,div=2000)
        with savepath.open("wb") as fp:
            torch.save(test_dataset,fp)
    else:
        with savepath.open("rb") as fp:
            test_dataset = torch.load(fp)

    fname = '../data/val_dataset_profile.pth'
    savepath = Path(fname)
    if not savepath.is_file():
        val_dataset = MyDataset(r"../data/val_data",cont_size = CONT_SIZE,div=2000)
        with savepath.open("wb") as fp:
            torch.save(test_dataset,fp)
    else:
        with savepath.open("rb") as fp:
            test_dataset = torch.load(fp)

    train_dataset.cont_size = CONT_SIZE
    test_dataset.cont_size = CONT_SIZE
    val_dataset.cont_size = CONT_SIZE

    train_dataset.div = 2000
    test_dataset.div = 2000
    val_dataset.div = 2000

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=my_collate,num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,collate_fn=my_collate,num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True,collate_fn=my_collate,num_workers=4)

    params = get_params(22,6,4,64,2,0.2)
    model = AttNet(*params,clf_dims=[512],embedding_dim=128)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,16,2)
    state = State(model,optim,scheduler)

    fname = "../models/state_Profile_SemiAxAtt.pth"


    _,_,_ = training_loop(train_dataloader, val_dataloader,fname=fname,epochs=4080,state=state,use_mut=False)
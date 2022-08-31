import torch
torch.cuda.empty_cache()
from profilehooks import profile
from math import sqrt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms 
import os, glob, time
import copy
import joblib, sys
import numpy as np
import scipy
from scipy import stats
from scipy import spatial
import os,sys, os.path
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.rdBase
from rdkit import DataStructs
from rdkit.DataStructs import BitVectToText
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import IPython
from IPython.core.display import SVG
from torch.autograd import Variable
import multiprocessing
import pandas as pd
from rdkit.Chem import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import rdkit
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import math
from numpy import sqrt
from numpy import argmax





#--------------------------------------------
##input file csv with smiles label
#----------------------------------------------


path="/rds/user/dh684/hpc-work/kekulescope/multitask/change_archi/output_dnn_CP_binary_mean_1696_30404.csv"#input file csv with smiles label and number
f=pd.read_csv(path).set_index("SMILES")
f_data=pd.read_csv(path).set_index("SMILES")
f.loc["label_1"]=(f==1).astype(int).sum(axis=0)
f.loc["label_0"]=(f==0).astype(int).sum(axis=0)
f.loc["dtff"]=f.loc["label_1"]/f.loc["label_0"]
max=f.loc["dtff"].max()
for i in f.columns:
    if f.loc['dtff'][i] == max:
        least=i
        
sss=StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=123)
ssss=StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=123)
f_data["number"]=range(len(f_data.index))
x = f_data
y = f_data[least]


for train_index, test_index in sss.split(x, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test_temp = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test_temp = y.iloc[train_index], y.iloc[test_index]
    
x2=x_test_temp
y2=x_test_temp[least]
 
for train_index, test_index in ssss.split(x2, y2):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_val, x_test = x.iloc[train_index], x.iloc[test_index]
    y_val, y_test = y.iloc[train_index], y.iloc[test_index]


my_smiles=list(f_data.index)
activities=f_data.iloc[:,:-1]
SMILES_ids=list(f_data["number"])
SMILES_ids = np.asarray(SMILES_ids)
activities =  np.asarray(activities)
my_smiles = np.asarray(my_smiles)

#--------------------------------------------
##divide into train,val,test
#----------------------------------------------
train_indices=np.asarray(x_train["number"])
val_indices=np.asarray(x_val["number"])
test_indices=np.asarray(x_test["number"])

activities_train = activities[train_indices]
activities_test = activities[test_indices]
activities_val = activities[val_indices]

SMILES_ids_train = SMILES_ids[train_indices]
SMILES_ids_test = SMILES_ids[test_indices]
SMILES_ids_val = SMILES_ids[val_indices]

my_smiles_train = my_smiles[train_indices]
my_smiles_test = my_smiles[test_indices]
my_smiles_val = my_smiles[val_indices]


#-----------------------------------------------
# generate png graph
#-----------------------------------------------
os.makedirs(path[:-4]+"/image/train/images",exist_ok=True)
os.makedirs(path[:-4]+"/image/val/images",exist_ok=True)
os.makedirs(path[:-4]+"/image/test/images",exist_ok=True)

n=0
for i in my_smiles_train:   
    mol = Chem.MolFromSmiles(i)
    img = Draw.MolToImage(mol)
    path_smile = path[:-4]+"/image/train/images/"+str(SMILES_ids_train[n])+".png"
    img.save(path_smile)
    n=n+1
    
n=0
for i in my_smiles_test:   
    mol = Chem.MolFromSmiles(i)
    img = Draw.MolToImage(mol)
    path_smile = path[:-4]+"/image/test/images/"+str(SMILES_ids_test[n])+".png"
    img.save(path_smile)
    n=n+1
    
n=0
for i in my_smiles_val:   
    mol = Chem.MolFromSmiles(i)
    img = Draw.MolToImage(mol)
    path_smile = path[:-4]+"/image/val/images/"+str(SMILES_ids_val[n])+".png"
    img.save(path_smile)
    n=n+1
    
    

transform = {
                'train': transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=90),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'val': transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=90),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'test': transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }



#------------------------------------
# Data loaders
#------------------------------------

from load_images import *


paths_labels_train=[]
for i,x in enumerate(activities_train):
        path_now = path[:-4]+"/image/train/images/"+str(SMILES_ids_train[i])+".png"
        now = (path_now , x)
        paths_labels_train.append(now)
        
paths_labels_val=[]
for i,x in enumerate(activities_val):
        path_now = path[:-4]+"/image/val/images/"+str(SMILES_ids_val[i])+".png"
        now = (path_now , x)
        paths_labels_val.append(now)
        
paths_labels_test=[]
for i,x in enumerate(activities_test):
        path_now = path[:-4]+"/image/test/images/"+str(SMILES_ids_test[i])+".png"
        now = (path_now , x)
        paths_labels_test.append(now)



workers=0
shuffle=False




trainloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels= paths_labels_train,
            transform=transform['train']),
            batch_size=8, shuffle=shuffle,
            num_workers=workers) 

valloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels= paths_labels_val,
            transform=transform['val']),
            batch_size=8, shuffle=shuffle,
            num_workers=workers) 

testloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels= paths_labels_test,
            transform=transform['test']),
            batch_size=8, shuffle=shuffle,
            num_workers=workers) 

dataloaders = {'train': trainloader, 'val':valloader, 'test':testloader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





#-----------------------------------------------
# Training the model
#-----------------------------------------------

use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() and x else torch.FloatTensor)
use_gpu()
batch_size=8
 
def train_model(model, criterion, optimizer, scheduler, num_epochs=70):
    start_time = time.time()
    model.cuda()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    early = 0
    all_loss_train=np.array([])
    all_loss_test=np.array([])
    for epoch in range(num_epochs):
        time_epoch = time.time()
        if epoch % 200 == 0:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)
        print('Epoch {}/{} {}'.format(epoch, num_epochs - 1, early))
        print('-' * 10)
        if early >= 250:
            break
        for phase in ['train', 'val']:
            epoch_loss=0.0
            deno=0.0
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()
            if phase == 'train': 
                pred=np.empty([0,activities.shape[1]])
                obs=np.empty([0,activities.shape[1]])
                num_batch=math.ceil((len(train_indices))/batch_size)
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    labels = labels.type(torch.FloatTensor)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        aa = time.time()
                        outputs = model(inputs)
                        preds=outputs
                        preds = preds.type(torch.FloatTensor)
                        preds=preds.sigmoid().squeeze()

                        loss_batch=0.0
                        preds1=preds.cpu().detach().numpy()
                        labels1=labels.cpu().detach().numpy()
                        print("len_train")
                        print(len(labels[0]))
                        for i in range(len(labels[0])):

                            loss_i = criterion(preds[:,i], labels.squeeze()[:,i])
                            loss_batch +=loss_i
                        loss = loss_batch/len(labels[0])                                              
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            aa = time.time()
                            loss.backward()
                            optimizer.step()
                            epoch_loss +=loss.detach().item()
                            pred = np.concatenate((pred,preds1),axis=0)
                            obs =np.concatenate((obs,labels1),axis=0)                                  
                    del inputs; del outputs;del preds ;del loss
                all_loss_train=np.append(all_loss_train,epoch_loss/num_batch) 
                              
                auc=[]
                threshold=[]    
                for i in range(activities.shape[1]):
                    pred_result=pred[:,i]
                    obs_result=obs[:,i]     
                    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obs_result, pred_result, pos_label=1)
                    gmeans = sqrt(tpr * (1-fpr))
                    ix = argmax(gmeans)
                    thres=thresholds[ix]
                    threshold.append(thres)
                    AUC = sklearn.metrics.auc(fpr, tpr)
                    auc.append(AUC)
               
                ff = open(path[:-4]+"/results/train_preds_{}_{}.txt".format(net,lr),'a')             
                for i,x in enumerate(pred):
                    ff.write("{}\t{}\n".format(obs[i], pred[i]))
                ff.write("train_auc:{}\n".format(auc))
                ff.write("train_threshold:{}\n".format(threshold))
                ff.close() 
                
                 
            if phase == 'val':
                pred=np.empty([0,activities.shape[1]])
                obs=np.empty([0,activities.shape[1]]) 
                num_batch=math.ceil((len(val_indices))/batch_size)                
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    labels = labels.type(torch.FloatTensor)
                    outputs = model(inputs)
                    preds=outputs
                    preds = preds.type(torch.FloatTensor) 
                    preds=preds.sigmoid()
                    loss_batch=0.0
                    preds1=preds.cpu().detach().numpy()
                    labels1=labels.cpu().detach().numpy()

                    for i in range(len(labels[0])):

                        loss_i = criterion(preds[:,i].squeeze(), labels[:,i].squeeze())
                        loss_batch +=loss_i
                    loss = loss_batch/len(labels[0])                                                              
                    epoch_loss +=loss.detach().item()
                    pred = np.concatenate((pred,preds1),axis=0)
                    obs =np.concatenate((obs,labels1),axis=0)                     

                    
                    del labels, outputs, inputs,preds,loss
                all_loss_test=np.append(all_loss_test,epoch_loss/num_batch)                                          

                print("val report")

                auc=[]
                threshold=[]    
                for i in range(activities.shape[1]):
                    pred_result=pred[:,i]
                    obs_result=obs[:,i]     
                    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obs_result, pred_result, pos_label=1)
                    gmeans = sqrt(tpr * (1-fpr))
                    ix = argmax(gmeans)
                    thres=thresholds[ix]
                    threshold.append(thres)
                    AUC = sklearn.metrics.auc(fpr, tpr)
                    auc.append(AUC)
 
                ff = open(path[:-4]+"/results/val_preds_{}_{}.txt".format(net,lr),'a')
                for i,x in enumerate(pred):
                    ff.write("{}\t{}\n".format(obs[i], pred[i]))
                ff.write("val_auc:{}\n".format(auc))
                ff.write("val_threshold:{}\n".format(threshold))
                ff.close()                
            print('{} Loss: {:.4f} {}'.format(phase,epoch,epoch_loss))    
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early=0
            if phase == 'val' and epoch_loss > best_loss:
                early+=1
        print('Epoch complete in {:.0f}m {:.0f}s'.format( (time.time() - time_epoch) // 60, (time.time() - time_epoch) % 60))
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))  
    plt.figure(figsize=(12,9))
    epoch=list(range(len(all_loss_train)))
    plt.plot(epoch,all_loss_train,color='blue')
    plt.plot(epoch,all_loss_test,color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(path[:-4]+"/results/loss.png")
    model.load_state_dict(best_model_wts)
    return model


#-----------------------------------------------
# Architectures
#-----------------------------------------------
net="alexnet_extended"
lr=0.01

if net == "alexnet": 
    model_ft = models.alexnet(pretrained=False)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(activities[0]))
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)
    model_ft = model_ft.to(device)

if net == "alexnet_extended": 
    model_ft = models.alexnet(pretrained=False)
    modules=[]
    modules.append( nn.Linear(in_features=9216, out_features=4096, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    # modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
    # modules.append( nn.ReLU(inplace=True) )
    # modules.append( nn.Dropout(p=0.5) )
    # modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
    # modules.append( nn.ReLU(inplace=True) )
    # modules.append( nn.Dropout(p=0.5) )
    # modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
    # modules.append( nn.ReLU(inplace=True) )
    # modules.append( nn.Dropout(p=0.5) )
    # modules.append( nn.Linear(in_features=100, out_features=len(activities[0]), bias=True) )
    # classi = nn.Sequential(*modules)
    # model_ft.classifier = classi
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)
    # model_ft = model_ft.to(device)
	modules.append( nn.Linear(in_features=4096, out_features=2048, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=2048, out_features=len(activities[0]), bias=True) )
    classi = nn.Sequential(*modules)
    model_ft.classifier = classi
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)#, momentum=0.95) #, nesterov=True)
    model_ft = model_ft.to(device)

if net == "densenet201": 
    model_ft = models.densenet201(pretrained=False)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(activities[0]))
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)#, momentum=0.95) #, nesterov=True)
    model_ft = model_ft.to(device)

if net == "densenet201_extended": 
    model_ft = models.densenet201(pretrained=False)
    modules=[]
    modules.append( nn.Linear(in_features=1920, out_features=4096, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=100, out_features=len(activities[0]), bias=True) )
    classi = nn.Sequential(*modules)
    model_ft.classifier = classi
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)
    model_ft = model_ft.to(device)

if net == "vgg19_bn": 
    model_ft = models.vgg19_bn(pretrained=False)
    modules=[]
    modules.append( nn.Linear(in_features=25088, out_features=4096, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=4096, out_features=2048, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=2048, out_features=len(activities[0]), bias=True) )
    classi = nn.Sequential(*modules)
    model_ft.classifier = classi
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)#, momentum=0.95) #, nesterov=True)
    model_ft = model_ft.to(device)

if net == "resnet152": 
    model_ft = models.resnet152(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(activities[0]))
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)#, momentum=0.95) #, nesterov=True)
    model_ft = model_ft.to(device)

if net == "resnet152_extended": 
    model_ft = models.resnet152(pretrained=False)
    modules=[]
    modules.append( nn.Linear(in_features=2048, out_features=4096, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
    modules.append( nn.ReLU(inplace=True) )
    modules.append( nn.Dropout(p=0.5) )
    modules.append( nn.Linear(in_features=100, out_features=len(activities[0]), bias=True) )
    classi = nn.Sequential(*modules)
    model_ft.fc = classi
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr)
    model_ft = model_ft.to(device)

criterion = torch.nn.BCELoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.6)




#-----------------------------------------------
# Training
#-----------------------------------------------
os.makedirs(path[:-4]+"/models",exist_ok=True)
os.makedirs(path[:-4]+"/results",exist_ok=True)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=70)

try:
    torch.save(model_ft, path[:-4]+'/models/model_{}_{}.pt'.format(net,lr))
    torch.save(model_ft.state_dict(), path[:-4]+'/models/model_dict_{}_{}.pt'.format(net,lr)) 
except:
    pass



#-----------------------------------------------
# Predictions test set
#-----------------------------------------------
model_ft.eval()

pred=np.empty([0,activities.shape[1]])
obs=np.empty([0,activities.shape[1]])


for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model_ft(inputs).sigmoid()
    
    pred = np.concatenate((pred,outputs.cpu().detach().numpy()),axis=0)
    obs =np.concatenate((obs,labels.cpu().detach().numpy()),axis=0)
    
auc=[]
threshold=[]    
for i in range(activities.shape[1]):
    pred_result=pred[:,i]
    obs_result=obs[:,i]     
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(obs_result, pred_result, pos_label=1)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    thres=thresholds[ix]
    threshold.append(thres)
    AUC = sklearn.metrics.auc(fpr, tpr)
    auc.append(AUC)
    plt.figure(figsize=(12,9))
    plt.plot(fpr,tpr,color='blue',label='Test ROC curve (area = %0.4f)' % AUC)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.savefig(path[:-4]+"/results/test_roc_curve_{}_{}.png".format(net,i))   


auc_test=pd.DataFrame(f.columns)
auc_test["auc"]=auc
auc_test.to_csv(path[:-4]+"/results/auc_test_{}.txt".format(net),'a')
ff = open(path[:-4]+"/results/preds_{}_{}.txt".format(net,lr),'a')
for i,x in enumerate(pred):
    ff.write("{}\t{}\n".format(obs[i], pred[i]))
ff.write("test_auc:{}\n".format(auc))
ff.write("test_threshold:{}\n".format(threshold))
ff.close()


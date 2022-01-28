import torch 
from torch.utils.data import Dataset
import torch.nn as nn 
from skimage import io 
import torch.nn.functional as F 
import numpy as np
import cv2 
import os 
import torchvision.transforms as T
import numpy as np 

class My_Dataset(Dataset): 

    def __init__(self,root,transform=None):
        self.root = root 
        imgs = [os.path.join(root,img) for img in os.listdir(root) if img !='.DS_Store'] 
        new_im = [os.path.join(i,img) for i in imgs for img in os.listdir(i)]
        np.random.seed(100)
        self.new_im = np.random.permutation(new_im)
        self.transform = transform
        
    def __len__(self): 
        return len(self.new_im)
    
    def __getitem__(self, index):
        dimension =64 
        img_path = self.new_im[index]
        label = 14
        name_obj = img_path.split('/')[-2]
        if 'Bedroom' in name_obj:
            label=0
        if 'Coast' in name_obj:
            label=1
        if 'Forest' in name_obj:
            label=2
        if 'Highway' in name_obj:
            label=3
        if 'Industrial' in name_obj:
            label=4
        if 'InsideCity' in name_obj:
            label=5
        if 'Kitchen' in name_obj:
            label=6
        if 'LivingRoom' in name_obj:
            label = 7
        if 'Mountain' in name_obj:
            label = 8
        if 'Office' in name_obj:
            label=9
        if 'OpenCountry' in name_obj:
            label = 10
        if 'Store' in name_obj:
            label = 11
        if 'Street' in name_obj:
            label = 12
        if 'Suburb' in name_obj:
            label = 13
        if 'TallBuilding' in name_obj:
            label =14
        image = io.imread(img_path)
        new_im =cv2.resize(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE),(dimension,dimension),interpolation=cv2.INTER_CUBIC)
        y_label = torch.tensor(label) 
        if self.transform: 
            new_im = self.transform(new_im) 
        
        return (new_im, y_label)
    
class Augmented_Dataset(Dataset): 
    def __init__(self,root,transform =None,dimension=64,augmentation=None,flip=None,crop=None,train=False,val=False):
        self.root = root
        imgs = [os.path.join(root,img) for img in os.listdir(root) if img !='.DS_Store'] 
        new_im = [os.path.join(i,img) for i in imgs for img in os.listdir(i) ]#if img !='.DS_Store']
        np.random.seed(100)
        self.new_im = np.random.permutation(new_im)
        
        if(train==True):
            self.new_im =self.new_im[0:int(0.85*len(self.new_im))]
        if(val==True): 
            self.new_im =self.new_im[int(0.85*len(self.new_im)):]
        #self.expand = expand         
        self.transform = transform
        self.augmentation = augmentation
        self.crop = crop 
        self.flip = flip 
        self.dimension = dimension
    
    def __len__(self): 
        return len(self.new_im)
    
    def __getitem__(self, index): 
        
        img_path = self.new_im[index]
        label = 14
        
        
        name_obj = img_path.split('/')[-2]
        
        if 'Bedroom' in name_obj:
            label=0
        if 'Coast' in name_obj:
            label=1
        if 'Forest' in name_obj:
            label=2
        if 'Highway' in name_obj:
            label=3
        if 'Industrial' in name_obj:
            label=4
        if 'InsideCity' in name_obj:
            label=5
        if 'Kitchen' in name_obj:
            label=6
        if 'LivingRoom' in name_obj:
            label = 7
        if 'Mountain' in name_obj:
            label = 8
        if 'Office' in name_obj:
            label=9
        if 'OpenCountry' in name_obj:
            label = 10
        if 'Store' in name_obj:
            label = 11
        if 'Street' in name_obj:
            label = 12
        if 'Suburb' in name_obj:
            label = 13
        if 'TallBuilding' in name_obj:
            label =14
        #image = io.imread(img_path)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if (self.augmentation!=None): 
            image = self.augmentation(image,self.flip,self.crop)
        new_im =cv2.resize(image,(self.dimension,self.dimension),interpolation=cv2.INTER_CUBIC)
        
        y_label = torch.tensor(label) 
        
      
           
        if self.transform: 
            new_im = self.transform(new_im) 
            
        return (new_im, y_label)   
    
    
def data_augmentation(image,flip=False, cropping = False):
    if(flip==True): 
        image = np.fliplr(image)
    if(cropping==True): 
        new_y = int(2/3*image.shape[0])
        new_x = int(2/3*image.shape[1])
        image = image[0:new_y,0:new_x]
        
    return image
            
    
        
        
        
        
def get_params_and_gradients_norm(named_parameters):
    square_norms_params = []
    square_norms_grads = []

    for _, param in named_parameters:
        if param.requires_grad:
            square_norms_params.append((param ** 2).sum())
            square_norms_grads.append((param.grad ** 2).sum())
    norm_params = sum(square_norms_params).sqrt().item()
    norm_grads = sum(square_norms_grads).sqrt().item()
    return norm_params, norm_grads

    
def mean_and_std(root = "train"): 
    dataset = My_Dataset(root = root, transform = T.ToTensor())
    x = 0
    x2 = 0
    for i in range(len(dataset)): 
        x += torch.sum(dataset[i][0][0][:][:])
        x2 += torch.sum((dataset[i][0][0][:][:])**2)
    mean = x /(len(dataset)*64*64)
    std = (x2/(len(dataset)*64*64) -mean**2)**(1/2)
    return mean, std
    

def train_model(model, num_epochs, train_loader, criterion, optimizer,writer,step = 0,trajectory=None, weights_check = False,device="cpu"): 
  
    for epoch in range(num_epochs): 
        step+=1
        losses = []
        accuracy =[]
        if(weights_check):
            writer.add_histogram('conv_1',model.conv_layers[0].weight, epoch)
            writer.add_histogram('bias_1',model.conv_layers[0].bias, epoch)
            writer.add_histogram('conv_2',model.conv_layers[3].weight,epoch)
            writer.add_histogram('bias_2',model.conv_layers[3].bias,epoch)
            writer.add_histogram('conv_3',model.conv_layers[6].weight,epoch)
            writer.add_histogram('bias_3',model.conv_layers[6].bias,epoch)
            writer.add_histogram('fcs',model.fcs[0].weight,epoch)
            writer.add_histogram('fcs_bias',model.fcs[0].bias,epoch)
        for _, (data,targets) in enumerate(train_loader): 
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores,targets)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            _, pred = scores.max(1) 
            num_correct = (pred==targets).sum() 
            running_train_acc = float(num_correct)/(float(data.shape[0]))
            accuracy.append(running_train_acc)
        mean_loss = np.mean(np.asarray(losses))
        mean_accuracy = np.mean(np.asarray(accuracy))
        writer.add_scalar('Training Loss',mean_loss,global_step =step)
        writer.add_scalar('Training_accuracy',running_train_acc,global_step=step)
        if (trajectory!=None): 
            
            params_norm, gradients_norm = get_params_and_gradients_norm(model.named_parameters())
            trajectory["parameters"].append(params_norm)
            trajectory["gradients"].append(gradients_norm)
            
def validation(model, val_loader, criterion,device="cpu"): 
    losses = []
    num_correct = 0
    num_samples = 0
    with torch.no_grad(): 
        for _, (data,targets) in enumerate(val_loader): 
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data) 
            loss = criterion(scores,targets)
            losses.append(loss.item())
            _, pred = scores.max(1)
            num_correct += (pred==targets).sum() 
            num_samples += pred.size(0) 
    
    mean_loss = np.mean(np.asarray(losses))
    accuracy = float(num_correct)/float(num_samples)*100
    return accuracy ,mean_loss
             
    
def early_stopping(PATH,first_epochs,model,train_loader,val_loader, criterion, optimizer,writer,max_epochs,patience=20,device="cpu"): 
    
    model.train() 

    step = 0
    train_model(model, first_epochs, train_loader, criterion, optimizer, writer,step,device=device)
    model.eval()

    accuracy_new,loss = validation(model, val_loader, nn.CrossEntropyLoss(),device=device)
    step+=first_epochs
    writer.add_scalar('Validation_accuracy',accuracy_new,step)
    writer.add_scalar('Validation_loss',loss,global_step=step)
    

   
    while(step<=max_epochs+1): #there is one more epoch to save the model
        
        
       
        torch.save({
            'parameters':model.state_dict()
        },PATH)
        
        accuracy_old = accuracy_new 
        count=0
        
        accuracy_new = 0
        
        while(accuracy_new<accuracy_old and count<patience): 
            count+=1
            
            
            model.train()
            train_model(model,1, train_loader, criterion, optimizer,writer,step,device=device)
            step+=1
            model.eval()
            accuracy_new, loss = validation(model, val_loader, criterion,device=device)
          
            writer.add_scalar('Validation_accuracy',accuracy_new,global_step=step)
            writer.add_scalar('Validation_loss',loss,global_step=step)
        
    
        if(count>=patience and accuracy_new<accuracy_old): 
            step = step -patience
           
            break 
        
            
    
    return step    
                 

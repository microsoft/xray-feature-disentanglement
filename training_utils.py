'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import time
import numpy as np

import torch
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        
        self.transform = transform
            
    def __getitem__(self, idx):
        
        img = self.images[idx]
        img = np.rollaxis(img, 0, 3)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]
                    
    def __len__(self):
        return self.images.shape[0]


class EmbeddingMultiTaskDataset(Dataset):
    
    def __init__(self, embeddings, label_sets):
        self.embeddings = embeddings
        self.label_sets = label_sets
        
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        embedding = torch.from_numpy(embedding)
        return embedding, [labels[idx] for labels in self.label_sets]
                    
    def __len__(self):
        return self.embeddings.shape[0]


class GradientReverse(torch.autograd.Function):
    '''https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/6'''
    scale = 1.0  
    @staticmethod 
    def forward(ctx, x): 
        return x.view_as(x)  
    @staticmethod 
    def backward(ctx, grad_output): 
        return GradientReverse.scale * grad_output.neg() 

def grad_reverse(x, scale=1.0): 
    GradientReverse.scale = scale 
    return GradientReverse.apply(x)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super(MultiTaskMLP, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        self.hidden_size = hidden_size
        self.num_tasks = len(output_sizes)
        self.heads = nn.Sequential(*[
            nn.Linear(hidden_size, output_size)
            for output_size in output_sizes
        ])
        
    def forward(self, x, scale=1.0):
        out = F.relu(self.hidden_layer(x))
        outs = []
        for i, head in enumerate(self.heads):
            if i == 0:
                outs.append(head(out))
            else:
                outs.append(head(grad_reverse(out,scale)))
        return outs


def fit(model, device, data_loader, optimizer, criterions, epoch, task_lr_multiplier, memo=''):
    model.train()
    
    losses = [
        []
        for i in range(len(criterions))
    ]
    tic = time.time()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        targets = [
            target.to(device)
            for target in targets
        ]
        
        optimizer.zero_grad()
        outputs = model(data, task_lr_multiplier)
        summed_loss = None
        for i, criterion in enumerate(criterions):
            loss = criterion(outputs[i], targets[i])
            losses[i].append(loss.item())
            if summed_loss is None:
                summed_loss = loss
            else:
                summed_loss = summed_loss + loss

        summed_loss.backward()
        optimizer.step()

    losses = np.array(losses).reshape(len(criterions), -1)
    losses = np.mean(losses, axis=1)

    print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Task Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, losses[0]), end=""
    )
    if len(losses) > 1:
        print("\t Domain Loss: {:.2f}".format(losses[1]), end="")
    print("")
    
    return losses

def evaluate(model, device, data_loader, criterions, epoch, memo=''):
    model.eval()
    
    losses = [
        []
        for i in range(len(criterions))
    ]
    tic = time.time()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        targets = [
            target.to(device)
            for target in targets
        ]
        
        with torch.no_grad():
            outputs = model(data)
            summed_loss = None
            for i, criterion in enumerate(criterions):
                loss = criterion(outputs[i], targets[i])
                losses[i].append(loss.item())
                if summed_loss is None:
                    summed_loss = loss
                else:
                    summed_loss = summed_loss + loss
    
    losses = np.array(losses).reshape(len(criterions), -1)
    losses = np.mean(losses, axis=1)

    print('[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Task Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, losses[0]), end=""
    )
    if len(losses) > 1:
        print("\t Domain Loss: {:.2f}".format(losses[1]), end="")
    print("")
    
    return losses

def score(model, device, data_loader, head_idx):
    model.eval()
    
    num_classes = model.heads[head_idx].out_features
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data)[head_idx])
        batch_size = data.shape[0]
        predictions[idx:idx+batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions

def embed(model, device, data_loader):
    model.eval()
    
    num_dimensions = model.hidden_size
    num_samples = len(data_loader.dataset)
    embeddings = np.zeros((num_samples, num_dimensions), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.relu(model.hidden_layer(data))
        batch_size = data.shape[0]
        embeddings[idx:idx+batch_size] = output.cpu().numpy()
        idx += batch_size
    return embeddings
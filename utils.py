'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import numpy as np
import pandas as pd

import cv2
import imageio

import torch
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchxrayvision as xrv

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = "./"

def get_binary_aucs(labels, prediction_probas, number_classes=3):
    all_aucs = []
    for class_idx in range(number_classes):
        mask = labels == class_idx
        binary_labels = np.zeros_like(labels)
        binary_labels[mask] = 1

        y_pred_proba = prediction_probas[:,class_idx]
        all_aucs.append(roc_auc_score(binary_labels, y_pred_proba, average=None)) 
    return all_aucs

def get_per_class_accuracies(labels, predictions, number_classes=3):
    all_accs = []
    for class_idx in range(number_classes):
        mask_true = labels == class_idx
        mask_predicted = predictions == class_idx
        
        num_correct = np.sum(mask_true & mask_predicted)
        num_total = np.sum(mask_true)
        
        all_accs.append(num_correct/num_total)
    return all_accs


#-------------------
# Helper methods for loading data from each of the datasets
#-------------------
def get_covidx_metadata():
    '''Note: we filter out duplicate patientids here'''
    metadata = pd.read_csv(os.path.join(BASE_DIR, "datasets/covidx/metadata.csv"))
    metadata.reset_index(inplace=True)
    idxs = metadata.groupby("patientid").first()["index"].sort_values().values # get the indices of the first rows of every group of patientids 
    metadata = metadata.loc[idxs]
    del metadata["index"]
    metadata.reset_index(drop=True, inplace=True)
    
    unmasked_image_fns = metadata["unmasked_image_path"].values
    masked_image_fns = metadata["masked_image_path"].values
    return unmasked_image_fns, masked_image_fns, metadata

def get_images(fns):
    '''Loads images from a list of filenames, ensures that there will be 3 channels in output'''
    num_samples = fns.shape[0]
    images = np.zeros((num_samples, 224, 224, 3), dtype=np.uint8)
    for i in range(num_samples):
        img = imageio.imread(os.path.join(BASE_DIR, fns[i]))
        if len(img.shape) == 3:
            assert img.shape[2] == 3
            assert np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,1] == img[:,:,2])
        elif len(img.shape) == 2:
            img = np.stack([
                img, img, img
            ], axis=2)
        else:
            raise Exception("Weird image shape")
        assert img.shape[0] == 224 and img.shape[1] == 224
        images[i] = img
    return images

def get_raw_covidx_images(masked=False):
    unmasked_image_fns, masked_image_fns, metadata = get_covidx_metadata()

    if masked:
        fns = masked_image_fns
    else:
        fns = unmasked_image_fns
    return get_images(fns)


#-------------------
# Misc methods
#-------------------
def transform_to_equalized(images):
    '''Apply histogram equalization to a stack of images'''
    images = images[:,:,:,0].copy()
    for i in range(images.shape[0]):
        images[i] = cv2.equalizeHist(images[i])
    images = np.stack([images,images,images], axis=3)
    return images


#-------------------
# Methods for normalizing images from [0,255] to format expected by a model
#-------------------
def transform_to_xrv(images):
    '''Used by torchxrayvision models'''
    images = images[:,:,:,0].astype(np.float32, copy=True)
    images = images[:,np.newaxis,:,:]
    for i in range(images.shape[0]):
        images[i,0] = xrv.datasets.normalize(images[i], 255.0)
    return images


def transform_to_standardized(images, masked=False):
    '''Used by densenet121 (and other types of densenet) torchvision models'''
    images = images.astype(np.float32, copy=True)
    images = np.rollaxis(images,3,1)

    if masked:
        mean = 37.7120
        std = 57.5283
    else:
        mean = 143.5170
        std = 58.5742

    images = (images - mean) / std
    return images

def transform_to_covidnet(images):
    '''Used by COVID-Net models'''
    images = images.astype(np.float32, copy=True)
    return images / 255.0


#-------------------
# Methods getting pre-trained model embeddings and dataset labels
#-------------------
def get_embeddings(dataset, mask, model):
    assert dataset in ["covidx"]
    assert model in ["xrv", "histogram-nozeros", "histogram", "densenet", "covidnet"]
    assert mask in ["masked", "unmasked"]

    fn = "%s_%s_%s.npy" % (dataset, mask, model)
    assert not os.path.exists(fn), "Generated embedding file does not exist"

    embeddings = np.load(os.path.join(BASE_DIR, "datasets/embeddings/", fn))

    return embeddings

def get_domain_labels(dataset):
    assert dataset in ["covidx"]

    _, __, metadata = get_covidx_metadata()
    dataset_names_to_idx_map = {
        'cohen':0, 'fig1':1, 'actmed':2, 'sirm':3, 'rsna':4
    }
    labels = metadata["dataset"].apply(lambda x: dataset_names_to_idx_map[x]).values

    return labels

def get_task_labels(dataset):
    assert dataset in ["covidx"]

    _, __, metadata = get_covidx_metadata()
    label_names_to_idx_map = {
        'normal': 0, 'pneumonia':1, 'COVID-19': 2
    }
    labels = metadata["label"].apply(lambda x: label_names_to_idx_map[x]).values

    return labels

def get_joint_ncp_embeddings_and_domain_labels(mask, model):
    assert model in ["xrv", "histogram-nozeros", "histogram", "densenet", "covidnet"]
    assert mask in ["masked", "unmasked"]

    covidx_embeddings = get_embeddings("covidx", mask, model)

    _, __, covidx_metadata = get_covidx_metadata()
    ncp_mask = (covidx_metadata["label"] == "COVID-19")
    covidx_embeddings = covidx_embeddings[ncp_mask]
    covidx_metadata = covidx_metadata[ncp_mask]
  
    covidx_datasets = ['cohen', 'fig1', 'actmed', 'sirm']
    covidx_labels = np.zeros((covidx_embeddings.shape[0]), dtype=np.int32)
    for i, label in enumerate(covidx_datasets):
        dataset_mask = covidx_metadata["dataset"] == label
        covidx_labels[dataset_mask] = i

    all_embeddings = np.concatenate([
        covidx_embeddings
    ], axis=0)

    all_labels = np.concatenate([
        covidx_labels,
    ], axis=0)
    all_labels = all_labels.astype(int)

    return all_embeddings, all_labels


#-------------------
# Methods for getting embeddings from images
#-------------------
def get_model_embedding_sizes(model):
    embedding_sizes = {
        "histogram": 256, "histogram-nozeros": 256, "xrv": 1024, "covidnet": 2048, "densenet": 1024
    }
    if model in embedding_sizes:
        return embedding_sizes[model]
    else:
        raise ValueError("%s is not recognized as a valid model" % (model))


def get_histogram_intensities(images, block_zero=False):
    '''Create pixel intensity histogram for each image in a stack of images'''
    num_samples = images.shape[0]
    assert images.dtype == np.uint8
    image_features = np.zeros((num_samples, 256), dtype=np.float32)
    for i in range(0, num_samples):
        image = images[i].ravel()
        vals = np.bincount(image, minlength=256)
        if block_zero:
            vals[0] = 0
        image_features[i] = vals / vals.sum()
    return image_features

def run_densenet_model(model, device, images, global_max_pool=False, embedding_size=1024, batch_size=128):
    num_samples = images.shape[0]
    image_embeddings = np.zeros((num_samples, embedding_size), dtype=np.float32)
    for i in range(0, num_samples, batch_size):
        image_batch = images[i:i+batch_size]
        with torch.no_grad():
            out = model.features(torch.from_numpy(image_batch).to(device))
        
            out = F.relu(out, inplace=True)
            if global_max_pool:
                out = F.adaptive_max_pool2d(out, (1, 1))
            else:
                out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = out.cpu().numpy().squeeze()
        image_embeddings[i:i+batch_size] = out
    return image_embeddings

def run_covidnet_model():
    # See `notebooks/Generate dataset embeddings with COVID-Net models.ipynb`
    raise NotImplementedError()

def run_xrv_model_features(model, device, images, batch_size=128):
    num_samples = images.shape[0]
    image_features = np.zeros((num_samples, 18), dtype=np.float32)
    for i in range(0, num_samples, batch_size):
        image_batch = images[i:i+batch_size]
        with torch.no_grad():
            output = model(torch.from_numpy(image_batch).to(device)).cpu().numpy()
        image_features[i:i+batch_size] = output
    return image_features


#-------------------
# Methods for instantiating pre-trained models 
#-------------------
def get_densenet121(device, num_classes=None, checkpoint=None):
    model = torchvision.models.densenet121(pretrained=True, progress=False)
    if num_classes is not None:
        model.classifier = nn.Linear(1024, num_classes)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model = model.eval().to(device)
    return model

def get_covidnet_model():
    # See `notebooks/Generate dataset embeddings with COVID-Net models.ipynb`
    raise NotImplementedError()

def get_xrv_model(device, num_classes=None, checkpoint=None):
    model = xrv.models.DenseNet(weights="all")
    if num_classes is not None:
        model.classifier = nn.Linear(1024, num_classes)
        model.op_threshs = None
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model = model.eval().to(device)
    return model
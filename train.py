'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import datetime
import argparse
import pickle
import copy

import numpy as np

import torch
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import utils
import training_utils

parser = argparse.ArgumentParser(description='X-ray fine-tuning script')
parser.add_argument('--split', type=int, required=True, help='Split number [0-9]')
parser.add_argument('--repetition', type=int, default=1, help='Repetition number')
parser.add_argument('--model', default='densenet',
    choices=["histogram", "histogram-nozeros", "xrv", "covidnet", "densenet"],
    help='Model embeddings to use'
)
parser.add_argument('--mask', default='unmasked',
    choices=(
        'unmasked',
        'masked'
    ),
    help='Choose between using unmasked or masked/equalized inputs'
)
parser.add_argument('--output_dir', type=str, required=True, help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to run on.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of training epochs')
parser.add_argument('--lr_start', type=float, default=1.0, help='Feature disentanglement learning rate starting point')
parser.add_argument('--hidden_layer_size', type=int, default=64, help='Size of the MLP model\'s hidden layer')
parser.add_argument('--use_feature_disentanglement', action="store_true", default=False, help='Enable training with feature disentanglement')
parser.add_argument('--save_embeddings', action="store_true", default=False, help='Save embeddings over all data')
args = parser.parse_args()


args.lr_exponent = 3.0
args.num_dataloader_workers = 4
device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')

def main():
    print("Starting x-ray fine-tuning script at %s" % (str(datetime.datetime.now())))

    assert args.split >= 0 and args.split <= 9, "Split number can only be in [0,9]."
    split_idx = args.split

    ## Ensure output directory exists
    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    #-------------------
    # Load data, setup device
    #-------------------
    all_embeddings = utils.get_embeddings("covidx", args.mask, args.model)
    all_task_labels = utils.get_task_labels("covidx")
    all_domain_labels = utils.get_domain_labels("covidx")


    #-------------------
    # Generate splits with same random numbers
    #-------------------
    np.random.seed(args.repetition)
    torch.manual_seed(args.repetition)

    train_splits = []
    val_splits = []
    test_splits = []

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.repetition)
    for i, (train_index, test_index) in enumerate(kf.split(all_embeddings, all_domain_labels)):
        
        train_index, val_index = train_test_split(train_index, test_size=0.1, stratify=all_domain_labels[train_index], random_state=args.repetition)

        train_splits.append(train_index)
        val_splits.append(val_index)
        test_splits.append(test_index)


    for i in range(len(train_splits)):
        assert len(np.unique(all_task_labels[train_splits[i]])) == 3
        assert len(np.unique(all_task_labels[val_splits[i]])) == 3
        assert len(np.unique(all_task_labels[test_splits[i]])) == 3

        assert len(np.unique(all_domain_labels[train_splits[i]])) == 5
        assert len(np.unique(all_domain_labels[val_splits[i]])) == 5
        assert len(np.unique(all_domain_labels[test_splits[i]])) == 5

    #-------------------
    # Setup datasets/dataloaders
    #-------------------
    scaler = StandardScaler()
    scaler = scaler.fit(all_embeddings[train_splits[split_idx]])
    all_embeddings = scaler.transform(all_embeddings)

    train_dataset = training_utils.EmbeddingMultiTaskDataset(
        all_embeddings[train_splits[split_idx]],
        [all_task_labels[train_splits[split_idx]], all_domain_labels[train_splits[split_idx]]],
    )
    val_dataset = training_utils.EmbeddingMultiTaskDataset(
        all_embeddings[val_splits[split_idx]],
        [all_task_labels[val_splits[split_idx]], all_domain_labels[val_splits[split_idx]]],
    )
    test_dataset = training_utils.EmbeddingMultiTaskDataset(
        all_embeddings[test_splits[split_idx]],
        [all_task_labels[test_splits[split_idx]], all_domain_labels[test_splits[split_idx]]]
    )


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
    )
    train_unshuffled_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )


    #-------------------
    # Training
    #-------------------
    mlp = training_utils.MultiTaskMLP(utils.get_model_embedding_sizes(args.model), args.hidden_layer_size, [3,5])
    mlp = mlp.to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    criterions = [
        nn.CrossEntropyLoss(), # task loss
    ]
    if args.use_feature_disentanglement:
        criterions.append(nn.CrossEntropyLoss()) # domain loss

    # Losses
    training_task_losses = []
    training_domain_losses = []
    validation_task_losses = []
    validation_domain_losses = []

    # AUCS
    validation_task_aucs = []
    validation_domain_aucs = []
    validation_task_accs = []
    validation_domain_accs = []

    # Other
    model_checkpoints = []

    num_times_lr_dropped = 0
    for epoch in range(args.num_epochs):
        lr = training_utils.get_lr(optimizer)
        
        task_lr = args.lr_start / (1+((args.lr_start-1)*(epoch/(args.num_epochs-1))**args.lr_exponent))
        
        training_losses = training_utils.fit(
            mlp,
            device,
            train_dataloader,
            optimizer,
            criterions,
            epoch,
            task_lr_multiplier=task_lr
        )
        
        validation_losses = training_utils.evaluate(
            mlp,
            device,
            val_dataloader,
            criterions,
            epoch
        )
        
        ## Record training/validation metrics
        training_task_losses.append(training_losses[0])
        validation_task_losses.append(validation_losses[0])
        if len(training_losses) > 1:
            training_domain_losses.append(training_losses[1])
            validation_domain_losses.append(validation_losses[1])

        ## Embed the entire training and validation sets
        train_set_embedding = training_utils.embed(mlp, device, train_unshuffled_dataloader)
        val_set_embedding = training_utils.embed(mlp, device, val_dataloader)
        
        ## Fit near optimal LR model on task labels
        task_lr_model = LogisticRegression(C=0.001, max_iter=20, random_state=args.repetition)
        task_lr_model.fit(train_set_embedding, all_task_labels[train_splits[split_idx]])
        y_pred_proba = task_lr_model.predict_proba(val_set_embedding)
        validation_task_auc = roc_auc_score(all_task_labels[val_splits[split_idx]], y_pred_proba, average="macro", multi_class="ovr")
        validation_task_acc = np.mean(utils.get_per_class_accuracies(all_task_labels[val_splits[split_idx]], y_pred_proba.argmax(axis=1)))
        print("Learned val task AUC:", validation_task_auc)
        validation_task_aucs.append(validation_task_auc)
        validation_task_accs.append(validation_task_acc)
        
        ## Fit near optimal LR model on domain labels 
        domain_lr_model = LogisticRegression(C=0.001, max_iter=20, random_state=args.repetition)
        domain_lr_model.fit(train_set_embedding, all_domain_labels[train_splits[split_idx]])
        y_pred_proba = domain_lr_model.predict_proba(val_set_embedding)
        validation_domain_auc = roc_auc_score(all_domain_labels[val_splits[split_idx]], y_pred_proba, average="macro", multi_class="ovr")
        validation_domain_acc = np.mean(utils.get_per_class_accuracies(all_domain_labels[val_splits[split_idx]], y_pred_proba.argmax(axis=1)))
        print("Learned val dataset AUC:", validation_domain_auc)
        validation_domain_aucs.append(validation_domain_auc)
        validation_domain_accs.append(validation_domain_acc)
        
        ## Copy near optimal LR model to model
        mlp.heads[1].weight.data = torch.from_numpy(domain_lr_model.coef_.astype(np.float32)).to(device)
        mlp.heads[1].bias.data = torch.from_numpy(domain_lr_model.intercept_.astype(np.float32)).to(device)
        
        model_checkpoints.append(copy.deepcopy(mlp.state_dict()))

        ## Early stopping
        scheduler.step(validation_losses[0])
        if training_utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")
        
        if num_times_lr_dropped == 3:
            break


    #-------------------
    # Testing
    #-------------------

    # Select best model
    if args.use_feature_disentanglement:
        best_model_checkpoint = model_checkpoints[np.argmin(validation_domain_aucs)]
    else:
        best_model_checkpoint = model_checkpoints[np.argmax(validation_task_aucs)]
    mlp.load_state_dict(best_model_checkpoint)


    # Evaluate on test tests
    y_pred_proba = training_utils.score(mlp, device, test_dataloader, 0)
    test_task_auc = roc_auc_score(all_task_labels[test_splits[split_idx]], y_pred_proba, average="macro", multi_class="ovr")
    test_task_acc = np.mean(utils.get_per_class_accuracies(all_task_labels[test_splits[split_idx]], y_pred_proba.argmax(axis=1)))
    print("Test task AUC:", test_task_auc)
    print("Test task ACC:", test_task_acc)
    print("")

    y_pred_proba = training_utils.score(mlp, device, test_dataloader, 1)
    test_domain_auc = roc_auc_score(all_domain_labels[test_splits[split_idx]], y_pred_proba, average="macro", multi_class="ovr")
    test_domain_acc = np.mean(utils.get_per_class_accuracies(all_domain_labels[test_splits[split_idx]], y_pred_proba.argmax(axis=1)))
    print("Test domain AUC:", test_domain_auc)
    print("Test domain ACC:", test_domain_acc)
    print("")


    # Save embeddings if we want to make UMAPs
    if args.save_embeddings:
        train_embedding = training_utils.embed(mlp, device, train_unshuffled_dataloader)
        val_embedding = training_utils.embed(mlp, device, val_dataloader)
        test_embedding = training_utils.embed(mlp, device, test_dataloader)

        all_embeddings = np.concatenate([
            train_embedding,
            val_embedding,
            test_embedding
        ], axis=0)


    #-------------------
    # Save everything
    #-------------------
    save_obj = {
        'args': args,
        'training_task_losses': training_task_losses,
        'training_domain_losses': training_domain_losses,
        'validation_task_losses': validation_task_losses,
        'validation_domain_losses': validation_domain_losses,
        'validation_task_aucs': validation_task_aucs,
        'validation_task_accs': validation_task_accs,
        'validation_domain_aucs':validation_domain_aucs,
        'validation_domain_accs': validation_domain_accs,
        "test_task_auc": test_task_auc,
        "test_task_acc": test_task_acc,
        "test_domain_auc": test_domain_auc,
        "test_domain_acc": test_domain_acc,
        "checkpoints": model_checkpoints
    }
    save_obj_fn = "covidx_%s_%s_%s_split-%d_%d_lr-%0.1f_hls-%d.pkl" % (args.mask, args.model, "disentangle" if args.use_feature_disentanglement else "no-disentangle",  args.split, args.repetition, args.lr_start, args.hidden_layer_size)
    with open(os.path.join(args.output_dir, save_obj_fn), 'wb') as f:
        pickle.dump(save_obj, f)

    if args.save_embeddings:
        save_embedding_fn = "covidx_%s_%s_%s_split-%d_%d_lr-%0.1f_hls-%d.npy" % (args.mask, args.model, "disentangle" if args.use_feature_disentanglement else "no-disentangle",  args.split, args.repetition, args.lr_start, args.hidden_layer_size)
        np.save(os.path.join(args.output_dir, save_embedding_fn), all_embeddings)


if __name__ == "__main__":
    main()


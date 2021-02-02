'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os, time
import datetime
import argparse
import pickle

import numpy as np
import pandas as pd

import torch
torch.backends.cudnn.deterministic = True

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from parse import parse

import utils
import training_utils

ALLOWED_FILENAMES = {'jpg', 'jpeg', 'png', 'dcm'}

parser = argparse.ArgumentParser(description='X-ray inference script')
parser.add_argument('--input_fn', type=str, required=True,  help='Path a "metadata.csv" file that has a `original_image_path` column with full paths to xray files')
parser.add_argument('--model_fn', type=str, required=True, help='Path to model file')
parser.add_argument('--output_fn', type=str, required=True,  help='Path to save output predictions (and possible embeddings) as npy. The containing directory must exist.')
parser.add_argument('--overwrite', action='store_true', default=False, help='Ignore checking whether the output directory has existing data')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to run on.')

parser.add_argument('--save_embeddings', action="store_true", default=False, help='Save embeddings over all data')
args = parser.parse_args()

device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')

def main():
    print("Starting x-ray inference script at %s" % (str(datetime.datetime.now())))

    # Parse out model arguments from the model filename -- we guarantee that our saved models will look like this
    assert os.path.exists(args.model_fn), "Model file does not exist"
    model_parts = parse(
        "covidx_{mask}_{model}_{disentangle}_split-{split}_{repetition}_lr-50.0_hls-64.pkl",
        os.path.basename(args.model_fn)
    )
    masked = model_parts["mask"] == "masked"
    base_model = model_parts["model"]
    assert base_model in ["xrv", "densenet"]
    disentangled = model_parts["disentangle"] == "disentangle"

    metadata_df = pd.read_csv(args.input_fn)
    if masked:
        original_fns = metadata_df["masked_image_path"].values
    else:
        original_fns = metadata_df["unmasked_image_path"].values
    num_samples = original_fns.shape[0]

    ## Ensure all input files exist
    for fn in original_fns:
        assert os.path.exists(fn), "Input doesn't exist: %s" % (fn)
        file_extension = fn.split(".")[-1]
        assert file_extension.lower() in ALLOWED_FILENAMES, "Input does not have a correct file extension: %s" % (file_extension)

    ## Ensure output directory exists
    output_dir = os.path.dirname(args.output_fn)
    if os.path.exists(output_dir):
        if os.path.exists(args.output_fn):
            if not args.overwrite:
                print("WARNING: The output file exists, exiting...")
                return
    else:
        os.makedirs(output_dir, exist_ok=True)


    ## Embed images with whatever the base model is
    tic = float(time.time())
    images = utils.get_images(original_fns) # these will be the masked versions if we are using a masked model
    print("Finished loading images in %0.4f seconds" % (time.time() - tic))

    tic = float(time.time())
    if masked:
        images = utils.transform_to_equalized(images)

    if base_model == "xrv":
        images = utils.transform_to_xrv(images)
        xrv_model = utils.get_xrv_model(device)

        embeddings = utils.run_densenet_model(
            xrv_model, device, images, global_max_pool=False, embedding_size=1024, batch_size=64
        )
    elif base_model == "densenet":
        images = utils.transform_to_standardized(images)
        densenet_model = utils.get_densenet121(device)
    
        embeddings = utils.run_densenet_model(
            densenet_model, device, images, global_max_pool=False, embedding_size=1024, batch_size=64
        )
    else:
        raise ValueError("Not implemented yet")

    ## Adjusting for normalization
    repetition = int(model_parts["repetition"])
    split_idx = int(model_parts["split"])
    all_embeddings = utils.get_embeddings("covidx", model_parts["mask"], base_model)
    all_task_labels = utils.get_task_labels("covidx")
    all_domain_labels = utils.get_domain_labels("covidx")
    np.random.seed(repetition)
    torch.manual_seed(repetition)
    train_splits = []
    val_splits = []
    test_splits = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=repetition)
    for i, (train_index, test_index) in enumerate(kf.split(all_embeddings, all_domain_labels)):
        train_index, val_index = train_test_split(train_index, test_size=0.1, stratify=all_domain_labels[train_index], random_state=repetition)
        train_splits.append(train_index)
        val_splits.append(val_index)
        test_splits.append(test_index)
    scaler = StandardScaler()
    scaler = scaler.fit(all_embeddings[train_splits[split_idx]])
    embeddings = scaler.transform(embeddings)

    test_dataset = training_utils.EmbeddingMultiTaskDataset(
        embeddings,
        [np.zeros(embeddings.shape[0])]
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=64,
        num_workers=1,
        pin_memory=True,
    )
    print("Finished embedding images in %0.4f seconds" % (time.time() - tic))

    
    ## Run embeddings through the saved model
    tic = float(time.time())
    with open(args.model_fn, "rb") as f:
        saved_model_params = pickle.load(f)

    mlp = training_utils.MultiTaskMLP(utils.get_model_embedding_sizes(base_model), saved_model_params["args"].hidden_layer_size, [3,5])
    mlp = mlp.to(device)

    if disentangled:
        best_model_checkpoint = saved_model_params["checkpoints"][np.argmin(saved_model_params["validation_domain_aucs"])]
    else:
        best_model_checkpoint = saved_model_params["checkpoints"][np.argmax(saved_model_params["validation_task_aucs"])]
    mlp.load_state_dict(best_model_checkpoint)

    test_set_pred_proba = training_utils.score(mlp, device, test_dataloader, 0)
    test_set_embedding = training_utils.embed(mlp, device, test_dataloader)
    print("Finished loading/running saved model in %0.4f seconds" % (time.time() - tic))

    # save output
    output_fn = args.output_fn
    if output_fn.endswith(".npy"):
        pass
    else:
        output_fn += ".npy"

    np.save(
        output_fn,
        test_set_pred_proba
    )
    
    if args.save_embeddings:
        output_fn = output_fn.replace(".npy", "_embeddings.npy")
        np.save(
            output_fn,
            test_set_embedding
        )


if __name__ == "__main__":
    main()
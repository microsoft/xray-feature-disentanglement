'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import datetime
import argparse

import numpy as np
import pandas as pd

import torch
torch.backends.cudnn.deterministic = True
import utils

parser = argparse.ArgumentParser(description='X-ray embedding script')
parser.add_argument('--model', default='densenet',
    choices=["histogram", "histogram-nozeros", "xrv", "covidnet", "densenet"],
    help='Type of embedding to create'
)
parser.add_argument('--mask', default='unmasked',
    choices=(
        'unmasked',
        'masked'
    ),
    help='Choose between using unmasked or masked/equalized inputs'
)

parser.add_argument('--input', type=str, required=True, help='Either "covidx" to use the covidx dataset, or a path to a "metadata_preprocessed.csv" file')
parser.add_argument('--name', type=str, required=True, help='Name of the dataset, this is used as the first field in the output filename')

parser.add_argument('--output_dir', type=str, default="datasets/embeddings/", help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--overwrite', action='store_true', default=False, help='Ignore checking whether the output file already exists')
parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to run on.')
args = parser.parse_args()


def run_covidnet_model(sess, image_tensor, pred_tensor, images, global_max_pool=False, embedding_size=2048, batch_size=128):
    num_samples = images.shape[0]
    image_embeddings = np.zeros((num_samples, embedding_size), dtype=np.float32)
    for i in range(0, num_samples, batch_size):
        image_batch = images[i:i+batch_size]
        out = sess.run(pred_tensor, feed_dict={image_tensor: image_batch})

        if global_max_pool:
            out = np.maximum(out, axis=(1,2))
        else:
            out = np.mean(out, axis=(1,2))
        image_embeddings[i:i+batch_size] = out
    return image_embeddings


def main():
    print("Starting x-ray embedding script at %s" % (str(datetime.datetime.now())))

    ## Ensure files aren't deleted and output directory exists
    output_fn = os.path.join(
        args.output_dir,
        f"{args.name}_{args.mask}_{args.model}.npy"
    )
    if os.path.exists(output_fn):
        if args.overwrite:
            print("WARNING: The output file already exists, but we are deleting that and moving on.")
        else:
            print("WARNING: The output file already exists and `--overwrite` was not specified, exiting...")
            return

    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir, exist_ok=True)


    ## Load imagery
    if args.input == "covidx":
        if args.mask == "unmasked":
            images = utils.get_raw_covidx_images(masked=False)
        elif args.mask == "masked":
            images = utils.get_raw_covidx_images(masked=True)
            images = utils.transform_to_equalized(images)
    else:
        df = pd.read_csv(args.input)
        if args.mask == "unmasked":
            images = utils.get_images(df["unmasked_image_path"].values)
        elif args.mask == "masked":
            images = utils.get_images(df["masked_image_path"].values)
            images = utils.transform_to_equalized(images)


    ## Embed imagery
    if args.model == "covidnet":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu is None else str(args.gpu)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        sess = tf.Session()
        tf.get_default_graph()
        saver = tf.train.import_meta_graph(os.path.join("data/pretrained_models/COVIDNet-CXR_Large/", "model.meta"))
        saver.restore(sess, os.path.join("data/pretrained_models/COVIDNet-CXR_Large/", "model-8485"))

        graph = tf.get_default_graph()

        image_tensor = graph.get_tensor_by_name("input_1:0")
        pred_tensor = graph.get_tensor_by_name("post_relu/Relu:0")

        images = utils.transform_to_covidnet(images)
        embeddings = run_covidnet_model(
            sess, image_tensor, pred_tensor, images, global_max_pool=False,
        )

    elif args.model == "xrv":
        device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')
        xrv_model = utils.get_xrv_model(device)
        images = utils.transform_to_xrv(images)
        embeddings = utils.run_densenet_model(
            xrv_model, device, images, global_max_pool=False, embedding_size=1024, batch_size=64
        )

    elif args.model == "densenet":
        device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')
        densenet_model = utils.get_densenet121(device)
        images = utils.transform_to_standardized(images)
        embeddings = utils.run_densenet_model(
            densenet_model, device, images, global_max_pool=False, embedding_size=1024, batch_size=64
        )

    elif args.model == "histogram":
        embeddings = utils.get_histogram_intensities(images)

    elif args.model == "histogram-nozeros":
        embeddings = utils.get_histogram_intensities(images, True)


    ## Write output
    np.save(output_fn, embeddings)


if __name__ == "__main__":
    main()


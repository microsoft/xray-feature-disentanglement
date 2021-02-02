'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import shutil
import datetime
import argparse
import uuid

import numpy as np
import pandas as pd

import imageio
import skimage.transform

import pydicom

ALLOWED_FILENAMES = {'jpg', 'jpeg', 'png', 'dcm'}

parser = argparse.ArgumentParser(description='X-ray preprocessing script')

parser.add_argument('--input_fn', type=str, required=True,  help='Path a "metadata.csv" file that has a `original_image_path` column with full paths to xray files')
parser.add_argument('--output_dir', type=str, required=True,  help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--overwrite', action='store_true', default=False, help='Ignore checking whether the output directory has existing data')
parser.add_argument('--gpu',  type=int, default=0,  help='ID of the GPU to run on.')
parser.add_argument('--disable_flip_preprocessing', action='store_true', default=False, help='Flag to disable the preprocessing step that checks heuristically whether an x-ray image represents the lightest pixel with "0", and if so, inverts values around the midpoint.')

args = parser.parse_args()

def main():
    print("Starting x-ray preprocessing script at %s" % (str(datetime.datetime.now())))


    #-------------------
    # Load data
    #-------------------
    metadata_df = pd.read_csv(args.input_fn)
    original_fns = metadata_df["original_image_path"].values
    num_samples = original_fns.shape[0]

    ## Ensure all input files exist
    for fn in original_fns:
        assert os.path.exists(fn), "Input doesn't exist: %s" % (fn)
        file_extension = fn.split(".")[-1]
        assert file_extension.lower() in ALLOWED_FILENAMES, "Input does not have a correct file extension: %s" % (file_extension)

    ## Ensure output directory exists
    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) > 0:
            if not args.overwrite:
                print("WARNING: The output directory is not empty, we are at risk of overwriting data, exiting...")
                return
            else:
                print("WARNING: The output directory is not empty, but we are deleting that and moving on.")
                shutil.rmtree(args.output_dir)
                os.makedirs(args.output_dir)

    else:
        os.makedirs(args.output_dir, exist_ok=True)
    

    #-------------------
    # Copy data to experiment folder and run preprocessing if necessary
    #-------------------
    original_images_dir = os.path.join(args.output_dir, "original_images/")
    lung_masks_dir = os.path.join(args.output_dir, "lung_masks/")
    lung_segmentation_dir = os.path.join(args.output_dir, "lung_segmented_images/")
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(lung_masks_dir, exist_ok=True)
    os.makedirs(lung_segmentation_dir, exist_ok=True)

    unmasked_img_fns = []
    base_fns = []
    for fn in original_fns:
        head, tail = os.path.split(fn)
        
        new_tail = "%s--%s" % (str(uuid.uuid4()), tail)
        file_extension = new_tail.split(".")[-1]
        if not file_extension == "jpg":
            new_tail = new_tail.replace("."+file_extension, ".jpg")
        output_fn = os.path.join(args.output_dir, "original_images/", new_tail)

        if tail.endswith(".dcm"): # convert dicom to jpg
            tail = tail + ".jpg"
            with pydicom.dcmread(fn) as f:
                img = f.pixel_array.copy()
                img = (img / img.max()).astype(np.float32)
                if f.PhotometricInterpretation == "MONOCHROME1":
                    img = 1 - img
                img = skimage.transform.resize(img, (224,224), clip=True)
                img = np.round(img * 255.0).astype(np.uint8)
                if len(img.shape) == 2:
                    img = np.stack([
                        img, img, img
                    ], axis=2)
                elif img.shape[2] == 1:
                    img = img.squeeze()
                    img = np.stack([
                        img, img, img
                    ], axis=2)
                imageio.imwrite(output_fn, img)
        else: # convert everything else to jpg
            img = imageio.imread(fn)
            img = skimage.transform.resize(img, (224, 224), clip=True) # this will put range into [0,1]

            if len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)
            elif len(img.shape) == 3:
                img = img[:,:,0].reshape(img.shape[0], img.shape[1], 1)

            if not args.disable_flip_preprocessing:
                hist, bin_edges = np.histogram(img[-20:,:,0].ravel(), bins=np.linspace(0,1,num=101))
                if np.sum(hist[:50]) < np.sum(hist[50:]):
                    pass # its fine
                else:
                    img = -img
            img = (img*255.0).astype(np.uint8)
            imageio.imwrite(output_fn, img)
        
        unmasked_img_fns.append(output_fn)
        base_fns.append(new_tail)
    metadata_df["unmasked_image_path"] = unmasked_img_fns


    #-------------------
    # Run lung segmentation VAE on all images 
    #-------------------
    os.system("python lungVAE/predict.py --model lungVAE/saved_models/lungVAE.pt --data %s --saveLoc %s" % (original_images_dir, lung_masks_dir))

    mask_fns = []
    masked_img_fns = []
    for i in range(num_samples):
        fn_no_extension = '.'.join(base_fns[i].split(".")[:-1])
        
        img_fn = unmasked_img_fns[i]
        mask_fn = os.path.join(lung_masks_dir, "%s_mask_post.png" % (fn_no_extension))
        masked_img_fn = os.path.join(lung_segmentation_dir, "%s_masked.png" % (fn_no_extension))
        
        mask_fns.append(mask_fn)
        masked_img_fns.append(masked_img_fn)

        img = imageio.imread(img_fn)
        mask = imageio.imread(mask_fn)

        masked_img = img.copy()
        masked_img[mask == 0] = 0

        imageio.imwrite(masked_img_fn, masked_img)

    metadata_df["masked_image_path"] = masked_img_fns

    #-------------------
    # Save the final results
    #-------------------
    metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)



if __name__ == "__main__":
    main()


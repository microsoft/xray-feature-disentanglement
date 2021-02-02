'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import pickle
import numpy as np

repetition_options = list(range(5))
model_options = ["xrv", "densenet", "covidnet"]
mask_options = ["unmasked", "masked"]
disentangle_options = ["no-disentangle", "disentangle"]
split_options = list(range(10))

print("Model test set performance on task and domain labels")
print("")
print("Feature extractor model,Masking method,Feature disentanglement method,Task label AUC,Domain label AUC")

for model in model_options:
    for mask in mask_options:
        for disentangle in disentangle_options:

            test_task_aucs = []
            test_task_accs = []
            test_domain_aucs = []
            test_domain_accs = []

            for repetition in repetition_options:
                for split in split_options:

                    fn = f"output/main_experiments/covidx_{mask}_{model}_{disentangle}_split-{split}_{repetition}_lr-50.0_hls-64.pkl"
                    if not os.path.exists(fn):
                        raise ValueError("Results file is missing")
                    with open(fn, "rb") as f:
                        checkpoint = pickle.load(f)

                    test_task_aucs.append(checkpoint["test_task_auc"])
                    test_task_accs.append(checkpoint["test_task_acc"])
                    test_domain_aucs.append(checkpoint["test_domain_auc"])
                    test_domain_accs.append(checkpoint["test_domain_acc"])

            print("%s,%s,%s,%0.2f +/- %0.2f,%0.2f +/- %0.2f" % (
                model, mask, disentangle,
                np.mean(test_task_aucs), np.std(test_task_aucs),
                np.mean(test_domain_aucs), np.std(test_domain_aucs),
            ))
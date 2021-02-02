'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import time
import itertools
import subprocess
from multiprocessing import Process, Queue

def do_work(work, gpu_idx):
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        subprocess.call(experiment.split(" "))
    return True

GPUS = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]

repetition_options = range(5)
model_options = ["xrv", "densenet", "covidnet"]
mask_options = ["masked", "unmasked"]
disentangle_options = ["disentangle", "no-disentangle"]
split_options = range(10)

work = Queue()

for repetition, model, mask, disentangle, split in itertools.product(repetition_options, model_options, mask_options, disentangle_options, split_options):

    command = f"python train.py --gpu GPU --split {split} --repetition {repetition} --model {model} --mask {mask} --lr_start 50.0 --output_dir output/main_experiments/"

    if disentangle == "disentangle":
        command += " --use_feature_disentanglement"

    work.put(command)

processes = []
start_time = time.time()
for gpu_idx in GPUS:
    p = Process(target=do_work, args=(work, gpu_idx))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
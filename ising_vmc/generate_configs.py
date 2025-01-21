import numpy as np 
from itertools import product
import json
import os
import shutil

optimizers = ['sgd', 'adam']
learnin_rates = [0.001, 0.01, 0.1]
system_size = [8]
alpha = [4]
symmetric = [True, False]
training_steps = [50,200]
dhs = [0.025, 0.05, 0.1]
h_low, h_high = [0.], [3.]


hyperparams = [system_size,
               optimizers,
               learnin_rates,
               alpha,   
               training_steps,
               symmetric,
               dhs,
               h_low,
               h_high]

hyperparams_names = ['system_size',
                     'optimizers',
                     'learning_rates',
                     'alpha',   
                     'training_steps',
                     'symmetric',
                     'dh',
                     'h_low',
                     'h_high']

folder_name = 'configs'
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, folder_name)
# remove folder first
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)


for i, comb in enumerate(product(*hyperparams)):

    config = {name: value for name, value in zip(hyperparams_names, comb)}    
    
    
    filename = os.path.join(output_folder, f'config_{i}.json')
    with open(filename, 'w') as json_file:
        json.dump(config, json_file, indent=4)


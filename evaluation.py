from tensorflow.keras.models import load_model
import numpy as np
import argparse
import h5py
import os
from layers.BilinearInterpolate3D import BilinearInterpolate3D
from layers.OrthogonalConv import OrthogonalConv

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", type=str, required=True,
                    help="Model with path")
parser.add_argument("-d", "--dataset", type=str, required=True,
                    help="Dataset to evaluate the model")
parser.add_argument("-b", "--batch_size", type=int, default=64,
                    help="Batch size")

# Parsing arguments from input command
args = parser.parse_args()
model_path = args.model
dataset_path = args.dataset
batch_size = args.batch_size

# Checking that given dataset and model files exist
if not os.path.exists(model_path): raise ValueError("Model specified does not exist. Please check path.")
if not os.path.exists(dataset_path): raise ValueError("Dataset specified does not exist. Please check path.")

# Import h5 dataset
with h5py.File(dataset_path, 'r') as h5File:
    if 'dataset' not in h5File.keys(): raise ValueError('h5 file has no dataset')
    dataset = h5File['dataset']
    x_set = np.array(dataset['x_set'])
    y_set = np.array(dataset['y_set'])

# Load model
model = load_model(model_path, custom_objects={'BilinearInterpolate3D': BilinearInterpolate3D, 
                                               'OrthogonalConv': OrthogonalConv})

# Evaluate model
scores = model.evaluate(x_set, y_set, batch_size=batch_size)

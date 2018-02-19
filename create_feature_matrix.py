import numpy as np
from os import path
from tqdm import tqdm
import json

features = []

labels_path = path.join('.', 'data', 'annotations', 'captions_train2014.json')
labels = json.load(open(labels_path))

for entry in tqdm(labels['images']):	
	save_path = path.join('.', 'data', 'train_features', entry['file_name'] + '.npz')
	features.append(np.load(save_path)['arr_0'].flatten())

features = np.stack(features, axis=0)
save_path = path.join('.', 'data', 'train_features_matrix.npz')
np.savez(save_path, features)
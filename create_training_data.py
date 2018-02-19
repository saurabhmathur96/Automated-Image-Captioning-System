from os import path
import json
import numpy as np
import pickle

max_caption_length = 16

captions_path = path.join('.', 'data', 'train_captions.json')
captions = json.load(open(captions_path))

index_path = path.join('.', 'data', 'index_to_token.json')
index_to_token = json.load(open(index_path))
# token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

labels_path = path.join('.', 'data', 'annotations', 'captions_train2014.json')
labels = json.load(open(labels_path))
images = labels['images']
image_id_to_index = {image['id']: i for i, image in enumerate(images)}


features_path = path.join('.', 'data', 'train_features_matrix.npz')
features = np.load(features_path)['arr_0']

image_id_to_features = {
    str(image['id']): feature_vector for image, feature_vector in zip(images, features)}


inputs = []
targets = []

for caption in captions:
    inputs.append(image_id_to_features[str(caption['image_id'])])

    caption = caption['token_indices']
    target = np.zeros(max_caption_length, dtype=int)
    target[0:len(caption)] = caption
    targets.append(target)

inputs = np.stack(inputs, axis=0)
targets = np.stack(targets, axis=0)

save_path = path.join('.', 'data', 'train_inputs.npz')
np.savez(save_path, inputs)
print ('input matrix of dimension %s saved at %s' %
       ('x'.join(map(str, inputs.shape)), save_path))

save_path = path.join('.', 'data', 'train_targets.npz')
np.savez(save_path, targets)
print ('target matrix of dimension %s saved at %s' %
       ('x'.join(map(str, targets.shape)), save_path))

# Read images and extract features using InceptionV3 trained
# on ImageNet dataset.

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from os import path
import numpy as np
import json
from tqdm import tqdm

inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(300, 359, 3), pooling='avg')

labels_path = path.join('.', 'data', 'annotations', 'captions_train2014.json')
labels = json.load(open(labels_path))
print (len(labels['images']))

for entry in tqdm(labels['images'][0:10], initial=0, total=len(labels['images'])):
	image_dir_path = path.join('.', 'data', 'train2014')
	image_path = path.join(image_dir_path, entry['file_name'])
	image_object = image.load_img(image_path, target_size=(300, 359))
	x = image.img_to_array(image_object)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = inception.predict(x)
	
	save_path = path.join('.', 'data', 'train_features', entry['file_name'] + '.npz')
	np.savez(save_path, features)
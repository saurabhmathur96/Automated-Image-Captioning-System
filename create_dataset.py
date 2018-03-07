from os import path
import json
import numpy as np
from tqdm import tqdm

max_caption_length = 16


captions_path = path.join('.', 'data', 'train_captions.json')
captions = json.load(open(captions_path))
# [{'id': 48,
# 'token_indices': [2, 54, 8349, 1519, 249, 8532, 2153, 2630, 622, 3],
# 'image_id': 318556}, ... ]


index_path = path.join('.', 'data', 'index_to_token.json')
index_to_token = json.load(open(index_path))


labels_path = path.join('.', 'data', 'annotations', 'captions_train2014.json')
labels = json.load(open(labels_path))
images = labels['images']
# [{'file_name': 'COCO_train2014_000000057870.jpg',
# 'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg',
# 'id': 57870, 
# 'height': 480, 
# 'license': 5, 
# 'date_captured': '2013-11-14 16:28:13',
# 'width': 640,
# 'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg'}, ...]
image_id_to_index = {image['id']: i for i, image in enumerate(images)}


input_image_feature_indices = []
input_partial_tokens = []
target_next_words = []

for caption in tqdm(captions):
    image_id = caption['image_id']
    image_index = image_id_to_index[image_id]
    token_indices = caption['token_indices']
    caption_length = len(token_indices)
    for i in range(caption_length):
        partial = np.zeros(max_caption_length-1, dtype=int)
        partial[0:i] = token_indices[:i]
        input_partial_tokens.append(partial)
        target_next_words.append(token_indices[i])
        input_image_feature_indices.append(image_index)


input_image_feature_indices = np.stack(input_image_feature_indices, axis=0)
input_partial_tokens = np.stack(input_partial_tokens, axis=0)
target_next_words = np.stack(target_next_words, axis=0)

save_path = path.join('.', 'data', 'input_image_feature_indices.npz')
np.savez(save_path, input_image_feature_indices)
print ('input matrix of dimension %s saved at %s' %
       ('x'.join(map(str, input_image_feature_indices.shape)), save_path))

save_path = path.join('.', 'data', 'input_partial_tokens.npz')
np.savez(save_path, input_partial_tokens)
print ('target matrix of dimension %s saved at %s' %
       ('x'.join(map(str, input_partial_tokens.shape)), save_path))


save_path = path.join('.', 'data', 'target_next_words.npz')
np.savez(save_path, target_next_words)
print ('target matrix of dimension %s saved at %s' %
       ('x'.join(map(str, target_next_words.shape)), save_path))
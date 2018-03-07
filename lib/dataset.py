import numpy as np
import random
import json
# from data_models import AnnotationInfo, CaptionDetails, ImageDetails, CaptionAnnotationData

class ImageFeatureCaptionDataGenerator(object):
    def __init__(self, image_features, input_image_feature_indices, input_partial_tokens, target_next_words, batch_size, maximum_caption_length, vocabulary_size, feature_vector_length):
        # Data
        self.image_features = image_features
        self.input_image_feature_indices = input_image_feature_indices
        self.input_partial_tokens = input_partial_tokens
        self.target_next_words = target_next_words
        
        # Meta-data and Bounds
        self.batch_size = batch_size
        self.maximum_caption_length = maximum_caption_length
        self.vocabulary_size = vocabulary_size
        self.feature_vector_length = feature_vector_length
        self.n_items = self.target_next_words.shape[0]
        
        # Start iterator index at zero
        self.indices = list(range(self.n_items))
        self.current_index = 0
        self.reset_index()

    def reset_index(self):
        self.current_index = 0
        random.shuffle(self.indices)

    def get_next_item(self):
        self.current_index += 1
        
        if self.current_index >= self.n_items:
            self.reset_index()

        i = self.indices[self.current_index]
        
        return self.image_features[self.input_image_feature_indices[i]], self.input_partial_tokens[i], self.target_next_words[i]

    def __iter__(self):
        return self

    def __next__(self):
        X_image = np.zeros((self.batch_size, self.feature_vector_length))
        X_partial = np.zeros((self.batch_size, self.maximum_caption_length-1))
        y = np.zeros((self.batch_size, self.vocabulary_size))
        for iteration in range(self.batch_size):
            current_x_image, current_x_partial, current_y = self.get_next_item()
            X_image[iteration, :] = current_x_image
            X_partial[iteration, :] = current_x_partial
            y[iteration, current_y] = 1
        inputs = {
            'image_feature_input': X_image,
            'partial_caption_input': X_partial
        }
        outputs = y
        return inputs, outputs

'''

def read_caption_annotation_data(file_path):
    dictionary = json.load(file_path)
    info, images, captions = dictionary['info'], dictionary['images'], dictionary['captions']
    info = AnnotationInfo(description = info['description'], url = info['url'],
        version = info['version'], year = info['year'],
        contributor = info['contributor'], date_created = info['date_created'])
    
    images = [ImageDetails(id = image['id'], license = image['license'],
        file_name = image['file_name'], height = image['height'], width = image['width'],
        date_captured = image['date_captured'], coco_url = image['coco_url'],
        flickr_url = image['flickr_url']) for image in images]
    
    captions = [CaptionDetails() for caption in captions]

    data = CaptionAnnotationData(info = info, images = images, captions = captions)
    return data
'''
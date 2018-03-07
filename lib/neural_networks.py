from keras.models import Model
from keras.layers import Dense, RepeatVector, LSTM, Bidirectional, Embedding, TimeDistributed, Activation, Input
from keras.layers.merge import concatenate 
from keras.optimizers import RMSprop
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
import numpy as np

def CaptionGeneratorModel(image_feature_vector_length, vocabulary_size, maximum_caption_length, hidden_size):

    image_model_input = Input(shape=(image_feature_vector_length,), name='image_feature_input')
    image_model_dense_output = Dense(hidden_size, activation='relu')(image_model_input)
    image_model_output = RepeatVector(maximum_caption_length-1)(image_model_dense_output)

    caption_model_input = Input(shape=(maximum_caption_length-1,), name='partial_caption_input')
    caption_model_embedding_output = Embedding(vocabulary_size, hidden_size*2, input_length=maximum_caption_length-1)(caption_model_input)
    caption_model_blstm_output = Bidirectional(LSTM(hidden_size, return_sequences=True))(caption_model_embedding_output)
    caption_model_output = TimeDistributed(Dense(hidden_size, activation='relu'))(caption_model_blstm_output)

    merged = concatenate([image_model_output, caption_model_output])
    blstm_output = Bidirectional(LSTM(hidden_size, return_sequences=False))(merged)
    dense_output = Dense(vocabulary_size)(blstm_output)
    output = Activation('softmax')(dense_output)
    final_model = Model(inputs=[image_model_input, caption_model_input], outputs=[output])

    return final_model

class InceptionV3FeatureExtractor:
    def __init__(self, image_shape, n_channels):
        self.image_shape = image_shape
        self.n_channels = n_channels
        input_shape = self.image_shape + [self.n_channels]
        self.model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    
    def extract_features(self, pixels):
        x = np.expand_dims(pixels, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features
    
    def get_model_summary(self):
        return self.model.summary()


    

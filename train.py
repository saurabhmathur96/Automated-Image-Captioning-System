from lib.dataset import ImageFeatureCaptionDataGenerator
from lib.neural_networks import CaptionGeneratorModel
from os import path
import numpy as np
import argparse
from datetime import datetime
from keras.callbacks import TensorBoard

if __name__ == '__main__':
    #
    # inputs  (391978, 2048)
    # targets (391978, 16)

    parser = argparse.ArgumentParser(description='Train the caption generation model')
    parser.add_argument('--batch_size', dest='batch_size',
                        default=32,  type=int)
    parser.add_argument('--maximum_caption_length',
                        dest='maximum_caption_length', default=16, type=int)
    parser.add_argument('--vocabulary_size',
                        dest='vocabulary_size', default=8769,  type=int)
    parser.add_argument('--feature_vector_length',
                        dest='feature_vector_length', default=2048,  type=int)
    parser.add_argument('--hidden_layer_size',
                        dest='hidden_layer_size', default=2048,  type=int)
    parser.add_argument('--epochs',
                        dest='epochs', default=5,  type=int)
    args = parser.parse_args()

    

    input_image_feature_indices_path = path.join('.', 'data', 'input_image_feature_indices.npz')
    input_image_feature_indices = np.load(input_image_feature_indices_path)['arr_0']

    input_partial_tokens_path = path.join('.', 'data', 'input_partial_tokens.npz')
    input_partial_tokens = np.load(input_partial_tokens_path)['arr_0']

    
    target_next_words_path = path.join('.', 'data', 'target_next_words.npz')
    target_next_words = np.load(target_next_words_path)['arr_0']

    image_features_path = path.join('.', 'data', 'train_features_matrix.npz')
    image_features = np.load(image_features_path)['arr_0']

    data_generator = ImageFeatureCaptionDataGenerator(image_features=image_features,
                                                      input_image_feature_indices=input_image_feature_indices,
                                                      input_partial_tokens = input_partial_tokens,
                                                      target_next_words=target_next_words,
                                                      batch_size=args.batch_size,
                                                      maximum_caption_length=args.maximum_caption_length,
                                                      vocabulary_size=args.vocabulary_size,
                                                      feature_vector_length=args.feature_vector_length)

    
    caption_model = CaptionGeneratorModel(image_feature_vector_length=args.feature_vector_length,
                                          vocabulary_size=args.vocabulary_size,
                                          maximum_caption_length=args.maximum_caption_length,
                                          hidden_size=128)
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
    caption_model.fit_generator(data_generator, steps_per_epoch=391978 // args.batch_size, epochs=args.epochs, callbacks=[tensorboard])
    save_path = path.join('..', 'models', 'model', 'word_model_%s.h5' % str(datetime.now()))
    caption_model.save(save_path)
from lib.dataset import ImageFeatureCaptionDataGenerator
from lib.neural_networks import CaptionGenerator
from os import path

if __name__ == '__main__':
    caption_model = CaptionGenerator(image_features_shape=(
        1024,), vocabulary_size=20000, maximum_caption_length=20, hidden_size=128)
    # data_generator = ImageFeatureCaptionDataGenerator(inputs=, targets=, batch_size=, maximum_caption_length=, vocabulary_size=, feature_vector_length=)
    
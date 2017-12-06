from keras.applications.inception_v3 import InceptionV3

inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(300, 359, 3), pooling='avg')

# Read images and extract features using InceptionV3 trained
# on ImageNet dataset.

print (inception.output_shape)
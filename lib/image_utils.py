from keras.preprocessing import image

def read_image_as_array(file_path, image_shape):
    # Read an image from filesystem and resize it.
    image_object = image.load_img(file_path, target_size=image_shape)
    pixels = image.img_to_array(image_object)
    return pixels
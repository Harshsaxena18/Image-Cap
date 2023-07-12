import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
token_path = 'saved_models/Flickr8k.token.txt'
train_images_path = 'saved_models/Flickr_8k.trainImages.txt'
test_images_path = 'saved_models/Flickr_8k.testImages.txt'
model_path = 'saved_models/Final_Image_Captioning.h5'

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def generate_caption(image_file):
    # Load the pre-trained model
    model = load_model(model_path)

    # Preprocess the image
    img = preprocess_image(image_file)

    # Generate the caption
    caption = generate_caption_from_image(model, img)

    return caption

def generate_caption_from_image(model, img):
    max_length = 34
    start_token = "<start>"
    end_token = "<end>"
    wordtoix = np.load("wordtoix.npy", allow_pickle=True).item()
    ixtoword = np.load("ixtoword.npy", allow_pickle=True).item()

    initial_state = [np.zeros((1, 256)), np.zeros((1, 256))]

    # Generate caption using greedy search
    caption = start_token
    for _ in range(max_length):
        sequence = [wordtoix[word] for word in caption.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([img, sequence] + initial_state)
        y_pred = np.argmax(y_pred)
        word = ixtoword[y_pred]
        caption += " " + word
        if word == end_token:
            break

    # Remove start and end tokens
    caption = " ".join(caption.split()[1:-1])

    return caption

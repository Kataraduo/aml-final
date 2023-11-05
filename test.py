import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
import json
import os
import argparse
from keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences


DATASET_RAW = "./datasets/flickr/raw"
DATASET_INTERIM = "./datasets/flickr/interim"
CHECKPOINTS = "./checkpoints/flickr"


def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except Exception as e:
        print(e)
        print(f"ERROR: Couldn't open image at {filename}")
        return
    image = image.resize((299, 299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, img_feature, max_length):
    in_text = 'start'
    caption = []
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Infer the next word
        pred = model.predict([img_feature, sequence], verbose=0)
        pred = np.argmax(pred)
        # Get the word from the tokenizer
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        caption.append(word)
        in_text += ' ' + word
        if word == 'end':
            break
    return ' '.join(caption)


if __name__ == "__main__":
    # Get the image index from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_idx = args['image']

    # Load the model and tokenizer
    max_length = 32
    tokenizer = load(
        open(os.path.join(DATASET_INTERIM, "tokenizer.pkl"), "rb")
    )
    model = load_model(os.path.join(CHECKPOINTS, 'baseline_9.h5'))
    xception_model = Xception(include_top=False, pooling="avg")

    # Load the image and generate the caption
    dataset_keys = json.load(
        open(os.path.join(DATASET_INTERIM, "dataset_keys.json"), "rb")
    )
    img_path = os.path.join(
        DATASET_RAW, 'Images', dataset_keys['test'][img_idx]
    )

    # Extract the image features
    img_feature = extract_features(img_path, xception_model)
    generated_caption = generate_caption(
        model, tokenizer, img_feature, max_length
    )

    # Print the caption and show the image
    print("\n")
    print(generated_caption)
    img = Image.open(img_path)
    plt.imshow(img)

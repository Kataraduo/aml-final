import numpy as np
import json
from pickle import load
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import get_captions, get_img_features

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


DATASET_INTERIM = "./datasets/flickr/interim"


class FlickrDataset():
    def __init__(
        self,
        max_length=32,
        batch_size=32,
        shuffle=True,
    ):
        "Initialization"
        self.max_length = max_length
        self.captions, self.features = self.load_feature_and_captions()
        self.tokenizer = self.load_tokenizer()
        self.batch_size = batch_size
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.shuffle = shuffle
        self.keys = list(self.captions.keys())
        self.indexes = np.arange(len(self.keys))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_feature_and_captions(self, mode="train"):
        dataset_keys = json.load(
            open(os.path.join(DATASET_INTERIM, "dataset_keys.json"), "rb")
        )
        captions = json.load(
            open(os.path.join(DATASET_INTERIM, "captions.json"), "rb"))
        img_features = load(
            open(os.path.join(DATASET_INTERIM, "img_features.pkl"), "rb")
        )

        captions = get_captions(captions, mode, dataset_keys)
        img_features = get_img_features(img_features, mode, dataset_keys)

        return captions, img_features

    def load_tokenizer(self):
        tokenizer = load(
            open(os.path.join(DATASET_INTERIM, "tokenizer.pkl"), "rb"))
        return tokenizer

    @property
    def size(self):
        return len(self.keys)

    def generate(self):
        while True:
            for key, caption_list in self.captions.items():
                feature = self.features[key][0]
                input_image, input_sequence, output_word = self.create_seq(
                    caption_list, feature)
                yield [input_image, input_sequence], output_word

    def create_seq(self, caption_list, feature):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in caption_list:
            # encode the sequence
            seq = self.tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                # encode output sequence
                out_seq = to_categorical(
                    [out_seq], num_classes=self.vocab_size)[0]
                # store
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

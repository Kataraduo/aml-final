import os
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


CHECKPOINTS = "./checkpoints/flickr"


class BaselineModel:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self.build_model()

    def build_model(self):
        # features from the CNN model squeezed from 2048 to 256 nodes
        # 优化1: Other image feature extraction methods (current: Xception)
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        # LSTM sequence model
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        # 优化2: Other decoders (e.g., Transformer)
        se3 = LSTM(256)(se2)

        # 第三个feature fe3
        # [1 0 0] - IG
        # [0 1 0] - LinkedIn
        # [0 0 1] - Existing dataset
           
        # Merging both models
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

        return model

    def train(self, generator, epochs, steps_per_epoch, verbose=1):
        # ([list of features, list of input sequences], list of output words)˝
        return self.model.fit(
            generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
        )

    def summary(self):
        return self.model.summary()

    def save(self, filename):
        self.model.save(os.path.join(CHECKPOINTS, filename))

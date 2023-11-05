from data.flickr_dataset import FlickrDataset
from models.baseline import BaselineModel
import os
import tensorflow as tf


def main():
    # Data generator
    data_generator = FlickrDataset()
    # Model
    model = BaselineModel(data_generator.vocab_size, data_generator.max_length)

    print('Dataset: ', data_generator.size)
    print('Vocabulary Size:', data_generator.vocab_size)
    print('Description Length: ', data_generator.max_length)
    print(model.summary())

    # Training
    epochs = 10
    steps = data_generator.size
    # making a directory models to save our models
    for i in range(epochs):
        generator = data_generator.generate()
        model.train(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save(f"baseline_{i}.h5")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()

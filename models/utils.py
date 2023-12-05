import tensorflow as tf


class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self, load_image, **kwargs):
        image_url = "https://tensorflow.org/images/surf.jpg"
        image_path = tf.keras.utils.get_file("surf.jpg", origin=image_url)
        self.image = load_image(image_path)

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()

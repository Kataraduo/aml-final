import pathlib
import collections
import tensorflow as tf
import re
import string
import einops
import tqdm


class Flickr8k:
    def __init__(
        self, path="flickr8k", image_shape=(224, 224, 3), vocabulary_size=5000
    ):
        self.path = pathlib.Path(path)
        self.image_shape = image_shape
        self.vocabulary_size = vocabulary_size
        self.train_raw, self.test_raw = self._load_data()
        self.tokenizer = self._setup_tokenizer()
        self.mobilenet = self._setup_mobilenet()

    def _load_data(self):
        captions = (self.path / "Flickr8k.token.txt").read_text().splitlines()
        captions = (line.split("\t") for line in captions)
        captions = ((fname.split("#")[0], caption) for (fname, caption) in captions)

        cap_dict = collections.defaultdict(list)
        for fname, cap in captions:
            cap_dict[fname].append(cap)

        train_files = (self.path / "Flickr_8k.trainImages.txt").read_text().splitlines()
        train_captions = [
            (str(self.path / "Images" / fname), cap_dict[fname])
            for fname in train_files
        ]

        test_files = (self.path / "Flickr_8k.testImages.txt").read_text().splitlines()
        test_captions = [
            (str(self.path / "Images" / fname), cap_dict[fname]) for fname in test_files
        ]

        train_ds = tf.data.experimental.from_list(train_captions)
        test_ds = tf.data.experimental.from_list(test_captions)

        return train_ds, test_ds

    def _setup_tokenizer(self):
        def standardize(s):
            s = tf.strings.lower(s)
            s = tf.strings.regex_replace(s, f"[{re.escape(string.punctuation)}]", "")
            s = tf.strings.join(["[START]", s, "[END]"], separator=" ")
            return s

        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocabulary_size, standardize=standardize, ragged=True
        )
        tokenizer.adapt(self.train_raw.map(lambda fp, txt: txt).unbatch().batch(1024))
        return tokenizer

    def _setup_mobilenet(self):
        mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape=self.image_shape, include_top=False, include_preprocessing=True
        )
        mobilenet.trainable = False
        return mobilenet

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_shape[:-1])
        return img

    def match_shapes(self, images, captions):
        caption_shape = einops.parse_shape(captions, "b c")
        captions = einops.rearrange(captions, "b c -> (b c)")
        images = einops.repeat(images, "b ... -> (b c) ...", c=caption_shape["c"])
        return images, captions

    def prepare_txt(self, imgs, txts):
        tokens = self.tokenizer(txts)
        input_tokens = tokens[..., :-1]
        label_tokens = tokens[..., 1:]
        return (imgs, input_tokens), label_tokens

    def prepare_dataset(self, ds, batch_size=32, shuffle_buffer=1000):
        ds = (
            ds.shuffle(10000)
            .map(lambda path, caption: (self.load_image(path), caption))
            .apply(tf.data.experimental.ignore_errors())
            .batch(batch_size)
        )

        def to_tensor(inputs, labels):
            (images, in_tok), out_tok = inputs, labels
            return (images, in_tok.to_tensor()), out_tok.to_tensor()

        return (
            ds.map(self.match_shapes, tf.data.AUTOTUNE)
            .unbatch()
            .shuffle(shuffle_buffer)
            .batch(batch_size)
            .map(self.prepare_txt, tf.data.AUTOTUNE)
            .map(to_tensor, tf.data.AUTOTUNE)
        )

    def save_dataset(self, ds, save_path, shards=10, batch_size=32):
        ds = (
            ds.map(lambda path, caption: (self.load_image(path), caption))
            .apply(tf.data.experimental.ignore_errors())
            .batch(batch_size)
        )

        def gen():
            for images, captions in tqdm.tqdm(ds):
                feature_maps = self.mobilenet(images)
                feature_maps, captions = self.match_shapes(feature_maps, captions)
                yield feature_maps, captions

        new_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=self.mobilenet.output_shape),
                tf.TensorSpec(shape=(None,), dtype=tf.string),
            ),
        )

        new_ds = new_ds.map(self.prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(1000)

        def shard_func(i, item):
            return i % shards

        new_ds.enumerate().save(save_path, shard_func=shard_func)

    def load_dataset(self, save_path, batch_size=32, shuffle=1000, cycle_length=2):
        def custom_reader_func(datasets):
            datasets = datasets.shuffle(1000)
            return datasets.interleave(lambda x: x, cycle_length=cycle_length)

        ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

        def drop_index(i, x):
            return x

        ds = (
            ds.map(drop_index, tf.data.AUTOTUNE)
            .shuffle(shuffle)
            .padded_batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

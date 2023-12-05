import tensorflow as tf
import matplotlib.pyplot as plt
import time

from data.flickr8k import Flickr8k
from models.attn_captioner import AttnCaptioner, TokenOutput
from models.utils import GenerateText


def visualize_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch #")
    plt.ylabel("CE/token")
    plt.grid()
    plt.legend()
    plt.savefig("loss.png")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["masked_acc"], label="accuracy")
    plt.plot(history.history["val_masked_acc"], label="val_accuracy")
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch #")
    plt.ylabel("CE/token")
    plt.grid()
    plt.legend()
    plt.savefig("accuracy.png")


def masked_loss(labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)

    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_acc(labels, preds):
    mask = tf.cast(labels != 0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
    return acc


def main():
    # Dataset
    flickr8k_dataset = Flickr8k(path="./datasets/flickr/raw")
    train_ds = flickr8k_dataset.prepare_dataset(flickr8k_dataset.train_raw)
    test_ds = flickr8k_dataset.prepare_dataset(flickr8k_dataset.test_raw)

    flickr8k_dataset.save_dataset(flickr8k_dataset.train_raw, "train_cache")
    flickr8k_dataset.save_dataset(flickr8k_dataset.test_raw, "test_cache")

    train_ds = flickr8k_dataset.load_dataset("train_cache")
    test_ds = flickr8k_dataset.load_dataset("test_cache")

    # Image feature extractor
    IMAGE_SHAPE = (224, 224, 3)
    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True
    )
    mobilenet.trainable = False

    # Model
    output_layer = TokenOutput(
        flickr8k_dataset.tokenizer, banned_tokens=("", "[UNK]", "[START]")
    )
    output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
    model = AttnCaptioner(
        flickr8k_dataset.tokenizer,
        feature_extractor=mobilenet,
        output_layer=output_layer,
        units=256,
        dropout_rate=0.5,
        num_layers=2,
        num_heads=2,
    )

    # Callbacks
    g = GenerateText()
    g.model = model
    g.on_epoch_end(0)

    callbacks = [
        GenerateText(),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=masked_loss,
        metrics=[masked_acc],
    )

    history = model.fit(
        train_ds.repeat(),
        steps_per_epoch=100,
        validation_data=test_ds.repeat(),
        validation_steps=20,
        epochs=100,
        callbacks=callbacks,
    )

    visualize_history(history)

    # Save the model to checkpoints/flickr8k named with the date and time
    model.save(
        f"checkpoints/flickr8k/{time.strftime('%Y%m%d-%H%M%S')}", save_format="tf"
    )


if __name__ == "__main__":
    main()

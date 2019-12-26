from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tfds.disable_progress_bar()
import numpy as np

print(tf.__version__)

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure.
    with_info=True)

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

for train_example, train_label in train_data.take(10):
    print('Encoded text:', train_example[:10].numpy())
    print('Label:', train_label.numpy())
    print('The originial review:\n', encoder.decode(train_example))

# Prepare the data for training
BUFFER_SIZE = 1000
'''
train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, train_data.output_shapes))

test_batches = (test_data.padded_bath(32, train_data.output_shapes))
'''

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, tf.compat.v1.data.get_output_shapes(train_data)))

test_batches = (
    test_data
    .padded_batch(32, tf.compat.v1.data.get_output_shapes(train_data)))

for example_batch, label_batch in train_batches.take(2):
    print('Batch shape:', example_batch.shape)
    print('Label shape:', label_batch.shape)

# Build the model
model = keras.Sequential([
    # This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
    # These vectors are learned as the model trains. The vectors add a dimension to the output array.
    # The resulting dimensions are: (batch, sequence, embedding).
    keras.layers.Embedding(encoder.vocab_size, 16),

    # This layer returns a fixed-length output vector for each example by averaging over the sequence dimension.
    # This allows the model to handle input of variable length, in the simplest way possible.
    keras.layers.GlobalAveragePooling1D(),

    # 1 single output node with the probability between 0 and 1.
    keras.layers.Dense(1, activation='sigmoid')])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Train the model
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30)

loss, accuracy = model.evaluate(test_batches)
print("Loss:", loss)
print("Accuracy", accuracy)

# Create a graph -----------------------------------------------------------------------------------------------------
# get a dictionary that contains everything during training
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for blue dots
plt.plot(epochs, loss, 'bo', label='Training loss')
# b for solid blue line
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()



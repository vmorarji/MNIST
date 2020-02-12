
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from ipywidgets import IntProgress


## using tensorflow 2.0.0-rc0
print(tf.__version__)


## load the MNIST dataset
mnist_dataset, mnist_info = tfds.load(name='mnist',with_info=True, as_supervised=True)


mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

## store both the number of of validation and test samples as an integer
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

## scale the data
def scale(image, label):
    image = tf.cast(image,tf.float32)
    image /=255.
    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)


BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

## create a validation set from 10% of the training data
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 150

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validations_targets = next(iter(validation_data))



## create the cnn
input_size = 784
output_size = 10
hidden_layer_size = 200

model = tf.keras.Sequential([
                            tf.keras.layers.Conv2D(32,(3, 3), activation='relu', input_shape=(28,28,1)),
                            tf.keras.layers.MaxPooling2D((2,2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                            tf.keras.layers.MaxPooling2D((2,2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
                            tf.keras.layers.Dense(output_size,activation='softmax')
                            ])




custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])




NUM_EPOCHS = 20
NUM_STEPS = num_validation_samples/BATCH_SIZE

# if the val_accuracy drops three times the training will stop
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

## train the model
model.fit(train_data,epochs=NUM_EPOCHS,callbacks=[early_stopping], validation_data=(validation_inputs,validations_targets),validation_steps=NUM_STEPS, verbose=2)


test_loss, test_accuracy = model.evaluate(test_data)


print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100))




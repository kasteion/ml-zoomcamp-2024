import tensorflow as tf
from tensorflow import keras

# Load the model using keras
model = keras.models.load_model('./clothing-model.keras')

# Save the model using tensorflow
tf.saved_model.save(model, 'clothing-model')
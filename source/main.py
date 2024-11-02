# Import TensorFlow into Colab
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import numpy as np
import pandas as pd

print("TF version:", tf.__version__)
print("TF Hub version:", hub.__version__)

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f'Loading saved model from {model_path}')

  #model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
  model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

  return model

# Create a function for preprocessing images
def process_image(image_path, img_size = 224):
  """
  Takes an image file path and turns the image into a Tensor.
  """
  # Read in an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0~255 to 0~1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired value (224,224)
  image = tf.image.resize(image, size=[img_size, img_size])

  return image

def create_data_batches(X, batch_size=32):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  print("Creating test data batches...")
  data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
  data_batch = data.map(process_image).batch(batch_size)
  return data_batch

def get_pred_label(prediction_probabilities):
  labels_csv = pd.read_csv('core/model/labels.csv')
  labels = labels_csv['breed'].to_numpy() # does same thing as above
  unique_breeds = np.unique(labels)

  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

loaded_model = load_model('core/model/dogVision.h5')

custom_predictions = loaded_model.predict(create_data_batches(['dog2.jpg', 'dog.jpg']), verbose=1)
for i in custom_predictions:
  custom_pred_labels = get_pred_label(i)
  print(custom_pred_labels)



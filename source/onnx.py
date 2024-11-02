import onnxruntime as ort
import numpy as np
import tensorflow as tf
import pandas as pd

def process_image(image_path, img_size=224):
    """
    圖片路徑轉換為數值化張量。
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    image = np.expand_dims(image, axis=0)  #增加batch維度
    return image

def get_pred_label(prediction_probabilities):
    labels_csv = pd.read_csv('core/model/labels.csv')
    labels = labels_csv['breed'].to_numpy()
    unique_breeds = np.unique(labels)
    return unique_breeds[np.argmax(prediction_probabilities)]

onnx_model_path = 'core/model/dogVision.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

image = process_image('dog.jpg')

input_name = ort_session.get_inputs()[0].name
predictions = ort_session.run(None, {input_name: image})

predicted_label = get_pred_label(predictions[0])
print(predicted_label)

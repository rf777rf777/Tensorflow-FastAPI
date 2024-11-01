# 將 Keras 模型轉換為 ONNX
import tf2onnx
import tensorflow as tf
import tensorflow_hub as hub

onnx_model_path = 'core/model/dogVision.onnx'

# 載入包含自訂層的模型
model_path = 'core/model/dogVision.h5'
keras_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})


# 將模型轉換為 ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13)
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model saved to {onnx_model_path}")

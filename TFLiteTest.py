import tensorflow as tf
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_save/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

temp = 100
interpreter.set_tensor(input_details[0]['index'], np.float32([[temp]]))

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]
print(output_data[0])

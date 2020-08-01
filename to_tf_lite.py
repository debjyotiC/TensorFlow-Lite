import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model('model_save/tf_keras_model')

tflite_model = converter.convert()

open("model_save/tflite_model/converted_model.tflite", "wb").write(tflite_model)

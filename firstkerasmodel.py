import numpy as np
import tensorflow as tf

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

l0 = tf.keras.layers.Dense(units=4, input_shape=[1], activation='relu')
l1 = tf.keras.layers.Dense(units=4, activation='relu')
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

model.save('model_save/tf_keras_model')

print(model.predict([[100]]))

#!/usr/bin/python3.9
#
# trains tensorflow model for PIMA Diabetes data
# mean squared error in whole data is 0.0616 in dinrhiw2 without overfitting
# and 0.001865 with overfitting and residual neural network (test_diabetes.sh)
#
# comparing training error when using TensorFlow
#
# tensorflow gives overfitting error 0.0020 in 100 epochs without or with residual neural network
# (Adam gradient descent). This means Adam gradient descent works a bit better
# and finds optimum faster. When running longer (1000 epochs) it gives 1e-7 error which
# is almost perfect memorization of training data.
#
# Tomas Ukkonen <tomas.ukkonen@iki.fi>
# 

import tensorflow as tf
import pandas as pd
import numpy as np

diabetes_train = pd.read_csv('diabetes.csv')
# names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
diabetes_features = diabetes_train.copy()
diabetes_labels = diabetes_features.pop('Outcome')
diabetes_features = np.array(diabetes_features)

normalize = tf.keras.layers.experimental.preprocessing.Normalization()
normalize.adapt(diabetes_features)

# 10 layer dense layers with ReLU activations (no residual neural networks)
diabetes_models = tf.keras.Sequential()
diabetes_models.add(tf.keras.layers.Input(shape=(8,)))
diabetes_models.add(normalize)
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(32, activation='relu'))
diabetes_models.add(tf.keras.layers.Dense(1))

# 10 layer dense layers with ReLU activations (residual neural network)
input = tf.keras.layers.Input(shape=(8,))
x = input
x = normalize(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
skip = x
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x + skip)
skip = x
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x + skip)
skip = x
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x + skip)
skip = x
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

diabetes_models = tf.keras.Model(inputs=input, outputs = x, name="ResNet")


diabetes_models.compile(loss = tf.losses.MeanSquaredError(),
                        optimizer = tf.optimizers.Adam())

diabetes_models.fit(diabetes_features, diabetes_labels, epochs=100)






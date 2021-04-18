import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 이미지의 밝고 어둡고의 정도를 0~255까지로 표현됨 #
#### /255 함으로써 Data를 0~1 사이의 값으로 Normalization

x_train = x_train / 255
x_test = x_test / 255

# 2. 이미지 데이터는 총 28 X 28 픽셀로 구성되어 있음
###### CNN Input은 28 X 28 좌표 상 0~1 사이의 1개의 Data를 지님
###### Input Shape = 28, 28, 1

# Reshape 이전
#### [X1,1, X1,2, .... , X1,28]
#### [X2,1, X2,2, .... , X2,28]
#### ..........................
#### [X28,1, X28,2, . , X28,28]

# Reshape 이후
#### [[X1,1], [X1,2], .... , [X1,28]]
#### [[X2,1], [X2,2], .... , [X2,28]]
#### ..........................
#### [[X28,1],[X28,2], .. , [X28,28]]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

learning_rate = 0.001
training_epochs = 10
batch_size = 128

model = keras.models.Sequential([

    keras.layers.Conv2D(32,7,activation="relu",padding="same",input_shape=[28,28,1]),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(32,3,activation="relu", padding="same"),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")

])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

y_predicted = model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

evaluation = model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
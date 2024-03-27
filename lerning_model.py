import tensorflow as tf
from tensorflow.keras import layers
import os
# # Загрузка и предобработка данных
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.your_dataset_here.load_data()

import tensorflow as tf
from tensorflow.keras.datasets import mnist
 

# Директория с сподготовленными данными цифры и буквы
dir1 = './data'

# Директория с сподготовленными фотографиями машин с номерами и выделенных машин
dir_plates = './DetectedPlates'

# Загрузка данных csv
file_data = 'data.csv'



# Загрузка данных 
(x_train, y_train), (x_test, y_test) = mnist.load_data() if os.path.exists(file_data) and os.path.getsize(dir1) and os.path.getsize(dir_plates) else mnist.load_data(path=file_data) 

# Предобработка данных
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
while True:
    # Определение модели CNN
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(36, activation="softmax")  # 36 классов (10 цифр + 26 букв)
    ])

    # Компиляция модели
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )



    # Обучение модели
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))




    # Определение модели CNN
    model1 = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(36, activation="softmax")  # 36 классов (10 цифр + 26 букв)
    ])

    # Компиляция модели
    model1.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )



    # Обучение модели
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    model1.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    # обучение заканчивается при достижении точности
    if model1.evaluate(x_test, y_test)[1] > 0.8 and model1.evaluate(x_test, y_test)[1] > 0.8 and not(model.evaluate(x_test, y_test)[1] != None): 
        break 









# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Точность 1:", test_acc)

test_loss, test_acc = model1.evaluate(x_test, y_test, verbose=2)
print("Точность 2:", test_acc)





# сохранение модели
model.save("model_resnet.tflite")

model.save("model1_resnet.tflite")

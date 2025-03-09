# ��Ȯ��:
# 0.9919
# 0.9912
# ����ð�: �� 17��

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam



now_time = time.time()
start_time = time.monotonic()

# MNIST �����ͼ� �ε�
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# �̹��� ũ�� ����ȭ
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# ������ ����ȭ
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Ŭ���� �� one-hot ���ڵ�
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# CNN �� ����
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# �� ������
learning_rate=0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# �� �н�
batch_size = 128
epochs = 200            # �����Լ��� ������ ����� 200��
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# �� ��
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ����ð� ���(2)
end_time = time.monotonic()
delta_time = time.time() - now_time

print("time: ", delta_time)
print("monotonic: ", end_time - start_time)

####################################
# �н� �սǰ� ���� �ս� �׷��� �׸���
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='x', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# �н� ��Ȯ���� ���� ��Ȯ�� �׷��� �׸���
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='x', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()


os.system("Pause")
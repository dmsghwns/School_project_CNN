# ��Ȯ��:
# 0.9755
# 0.97
# ����ð�: ��� 30��

from os import system
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

now_time = time.time()
start_time = time.monotonic()

# MNIST �����ͼ� �ε�
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ������ ��ó��
x_train = x_train / 255.0
x_test = x_test / 255.0

# �� ����
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# �� ������
learning_rate=0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# �� �н�
history = model.fit(x_train, tf.one_hot(y_train, 10),
          epochs=200,                               # �����Լ��� ������ ����� 200��
          batch_size=32,
          validation_data=(x_test, tf.one_hot(y_test, 10)))

# �� ��
loss, accuracy = model.evaluate(x_test, tf.one_hot(y_test, 10))
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

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

system("Pause")
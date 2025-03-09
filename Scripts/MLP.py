# 정확도:
# 0.9755
# 0.97
# 실행시간: 평균 30초

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

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

# 모델 정의
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
learning_rate=0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, tf.one_hot(y_train, 10),
          epochs=200,                               # 이차함수의 극점에 가까운 200번
          batch_size=32,
          validation_data=(x_test, tf.one_hot(y_test, 10)))

# 모델 평가
loss, accuracy = model.evaluate(x_test, tf.one_hot(y_test, 10))
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 실행시간 출력(2)
end_time = time.monotonic()
delta_time = time.time() - now_time

print("time: ", delta_time)
print("monotonic: ", end_time - start_time)

####################################
# 학습 손실과 검증 손실 그래프 그리기
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='x', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# 학습 정확도와 검증 정확도 그래프 그리기
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='x', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

system("Pause")
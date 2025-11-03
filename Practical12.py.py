import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("📘 CIFAR-10 Dataset Loaded Successfully!")
print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Training the CNN Model...")
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"True:{y_true[i]}\nPred:{y_pred_classes[i]}")
    plt.axis('off')
plt.show()

print("\n🎯 Outcome: CNN successfully trained on CIFAR-10 and made image predictions.")

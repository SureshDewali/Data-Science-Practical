import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print("Name: Suresh Dewali")
print("Roll No: 1323575\n")

# Step 1: Load Dataset (MNIST - Handwritten Digits)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("📘 MNIST Dataset Loaded Successfully!")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 2: Preprocess the Data
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Step 3: Build the Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),        
    Dense(128, activation='relu'),        
    Dense(64, activation='relu'),         
    Dense(10, activation='softmax')       
])

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
print("\n🚀 Training the Neural Network...")
history = model.fit(X_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# Step 7: Visualize Training Accuracy and Loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("\n🎯 Outcome: A simple ANN was built, trained, and evaluated on the MNIST dataset successfully.")

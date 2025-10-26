# cnn_model.py
# Author: Riddhi
# Project: CNN Image Classification with CIFAR-10 + Custom Images

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os, glob

# -------------------------------
# 1. Load and preprocess CIFAR-10
# -------------------------------
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print(f"✅ Data loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")

# -------------------------------
# 2. Build the CNN model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# -------------------------------
# 3. Compile and train
# -------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# -------------------------------
# 4. Evaluate on test data
# -------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# -------------------------------
# 5. Save accuracy plot
# -------------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/accuracy_plot.png')
plt.close()
print("📊 Saved: results/accuracy_plot.png")

# -------------------------------
# 6. Predict on your own images
# -------------------------------
img_paths = sorted(glob.glob("my_images/*.*"))

if len(img_paths) == 0:
    print("⚠️ No images found in 'my_images/'. Add JPG/PNG files to test your model.")
else:
    preds_text = []
    imgs_to_show = []

    for p in img_paths:
        try:
            # Load image and resize to 32x32
            img = image.load_img(p, target_size=(32, 32))
            arr = image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            # Predict class
            pred = model.predict(arr, verbose=0)
            idx = np.argmax(pred, axis=1)[0]
            label = class_names[idx]
            preds_text.append((os.path.basename(p), label))
            imgs_to_show.append((np.squeeze(arr, axis=0), label, os.path.basename(p)))
        except Exception as e:
            print(f"Error processing {p}: {e}")

    # Print predictions
    print("\n✅ Predictions on your custom images:")
    for fname, lbl in preds_text:
        print(f"  {fname:>20}  ->  {lbl}")

    # Create a collage of predictions
    cols = min(5, len(imgs_to_show))
    rows = int(np.ceil(len(imgs_to_show) / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i, (img_arr, lbl, fname) in enumerate(imgs_to_show, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(img_arr)
        plt.title(f"{lbl}\n({fname})", fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/my_predictions.png")
    plt.close()
    print("📁 Saved collage: results/my_predictions.png")

print("\n✅ All done! Check the 'results' folder for output images.")

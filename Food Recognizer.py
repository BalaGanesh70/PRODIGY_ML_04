import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the dataset
def load_dataset(data_dir):
    classes = sorted(os.listdir(data_dir))
    images = []
    labels = []
    calorie_content = []

    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_path, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(i)
                # Replace with code to extract calorie content from image metadata or a separate file
                calorie_content.append(extract_calorie_content_from_metadata(img_path))

    return np.array(images), np.array(labels), np.array(calorie_content)

# Replace with the path to your dataset
dataset_path = "path/to/food_dataset"
X, y, calorie_content = load_dataset(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, calorie_train, calorie_test = train_test_split(
    X, y, calorie_content, test_size=0.2, random_state=42
)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the model
image_input = layers.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(len(np.unique(y)), activation='softmax')(x)

model = models.Model(inputs=image_input, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model for food item recognition
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Food Item Recognition Accuracy: {accuracy}")

# Build a regression model for calorie estimation
calorie_input = layers.Input(shape=(1,))
x = layers.concatenate([model.output, calorie_input])
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
calorie_output = layers.Dense(1, activation='linear')(x)

calorie_model = models.Model(inputs=[image_input, calorie_input], outputs=calorie_output)

# Compile the calorie estimation model
calorie_model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the calorie estimation model
calorie_model.fit([X_train, calorie_train], calorie_train, epochs=10, validation_data=([X_test, calorie_test], calorie_test))

# Evaluate the model for calorie estimation
calorie_pred = calorie_model.predict([X_test, calorie_test])
mae = mean_absolute_error(calorie_test, calorie_pred)
print(f"Calorie Estimation Mean Absolute Error: {mae}")

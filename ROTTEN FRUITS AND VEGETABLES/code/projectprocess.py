import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Use a raw string for Windows path
dataset_dir =  r'C:\Users\ramad\OneDrive\Desktop\ROTTEN FRUITS AND VEGETABLES\images'

# Check directory and class folders
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"The dataset directory does not exist: {dataset_dir}")

class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
if not class_folders:
    raise FileNotFoundError(f"No class subfolders found in {dataset_dir}. Each class must be in its own subfolder.")

print("Found classes:", class_folders)

# Use a smaller validation split if you have very few images
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=4,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=4,
    class_mode='categorical',
    subset='validation'
)

# Check that data generators are not empty
if train_gen.samples == 0:
    raise ValueError("No training images found. Check your dataset directory and file types.")
if val_gen.samples == 0:
    print("Warning: No validation images found. Training will proceed without validation.")
    val_gen = None

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_folders), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if val_gen:
    model.fit(train_gen, epochs=10, validation_data=val_gen)
else:
    model.fit(train_gen, epochs=10)

os.makedirs('trained_model', exist_ok=True)
model.save('trained_model/trained_model.h5')

os.makedirs('labels', exist_ok=True)
with open('labels/labels.txt', 'w') as f:
    for label in train_gen.class_indices:
        f.write(label + '\n')
# Lung-infection
Build a model using a convolutional neural network that can classify lung infection in a person using medical imagery

```python
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.applications import MobileNet
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report

# Set the path to your train and test data folders
train_data_dir = r"C:\Users\prana\Desktop\courses\purdue AI & ML\data\Dataset_Detection_of_Lung_Infection\data\test"
test_data_dir = r"C:\Users\prana\Desktop\courses\purdue AI & ML\data\Dataset_Detection_of_Lung_Infection\data\train"

# Define the input image dimensions
img_width, img_height = 48, 48
input_shape = (img_width, img_height, 3)  # RGB images

# Set other parameters
batch_size = 32
epochs = 10  # Number of epochs to train for
patience = 2  # Early stopping patience

# Initialize an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load the training and testing datasets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the function to plot sample images
def plot_sample_images(image_generator, title):
    plt.figure(figsize=(10, 10))
    for images, labels in image_generator:
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(labels[i])
            plt.axis('off')
        break
    plt.suptitle(title, fontsize=16)
    plt.show()

# Plot sample images for training and testing datasets
plot_sample_images(train_generator, 'Sample Training Images')
plot_sample_images(test_generator, 'Sample Test Images')


# Plot sample images for each class
def plot_sample_images_per_class(image_generator, title):
    class_names = list(image_generator.class_indices.keys())
    plt.figure(figsize=(15, 8))
    for class_index, class_name in enumerate(class_names):
        plt.subplot(1, len(class_names), class_index + 1)
        class_images = [os.path.join(train_data_dir, class_name, img_name) for img_name in os.listdir(os.path.join(train_data_dir, class_name))][:3]
        for img_path in class_images:
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title(class_name)
            plt.axis('off')
            break
    plt.suptitle(title, fontsize=16)
    plt.show()

# Plot sample images for each class in the training dataset
plot_sample_images_per_class(train_generator, 'Sample Images per Class in Training Dataset')

# Plot the distribution of images across classes
def plot_class_distribution(image_generator, title):
    class_counts = image_generator.classes
    class_names = list(image_generator.class_indices.keys())
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, np.bincount(class_counts))
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.show()

# Plot the distribution of images across classes in the training dataset
plot_class_distribution(train_generator, 'Distribution of Images across Classes in Training Dataset')


# Initialize an ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Define the input image dimensions
img_width, img_height = 48, 48

# Load the training and testing datasets with data augmentation and preprocessing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


# Example code snippet for fetching a batch of images and labels during training
batch_images, batch_labels = next(train_generator)
# Example code snippet for fetching a batch of test images and labels during evaluation
test_batch_images, test_batch_labels = next(test_generator)

# Function to build a CNN model
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_generator.class_indices), activation='softmax'))
    return model

# Build and compile the CNN model
cnn_model = build_cnn_model()
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

# Train the CNN model using data generators
cnn_history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Plot training and validation accuracy, as well as loss
def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.show()

# Plot training and validation history for the CNN model
plot_training_history(cnn_history, 'CNN Model Training and Validation History')


from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D

# Load the pre-trained MobileNet model
base_mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers on top of the pre-trained model
x = base_mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the full model
mobilenet_model = Model(inputs=base_mobilenet.input, outputs=predictions)

# Compile the model
mobilenet_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the MobileNet-based model
mobilenet_history = mobilenet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Plot training and validation history for the MobileNet-based model
plot_training_history(mobilenet_history, 'MobileNet-Based Model Training and Validation History')


from keras.applications import DenseNet121

# Load the pre-trained Densenet121 model
base_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers on top of the pre-trained model
x = base_densenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the full model
densenet_model = Model(inputs=base_densenet.input, outputs=predictions)

# Compile the model
densenet_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Densenet121-based model
densenet_history = densenet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Plot training and validation history for the Densenet121-based model
plot_training_history(densenet_history, 'Densenet121-Based Model Training and Validation History')


# Function to evaluate model performance and print classification report
def evaluate_model_performance(model, generator):
    class_names = list(generator.class_indices.keys())
    true_labels = []
    predicted_labels = []

    for i in range(len(generator)):
        batch_images, batch_labels = next(generator)
        predictions = model.predict(batch_images)
        
        true_labels.extend(np.argmax(batch_labels, axis=1))
        predicted_labels.extend(np.argmax(predictions, axis=1))

    classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names)
    print(classification_rep)

# Evaluate CNN model performance
print("CNN Model Performance:")
evaluate_model_performance(cnn_model, test_generator)

# Evaluate MobileNet-based model performance
print("MobileNet-Based Model Performance:")
evaluate_model_performance(mobilenet_model, test_generator)

# Evaluate Densenet121-based model performance
print("Densenet121-Based Model Performance:")
evaluate_model_performance(densenet_model, test_generator)


from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D

# Set the path to your train and test data folders
train_data_dir = r"C:\Users\prana\Desktop\courses\purdue AI & ML\data\Dataset_Detection_of_Lung_Infection\data\test"
test_data_dir = r"C:\Users\prana\Desktop\courses\purdue AI & ML\data\Dataset_Detection_of_Lung_Infection\data\train"

# Define the input image dimensions
img_width, img_height = 224, 224  # MobileNet input size

# Set other parameters
batch_size = 32
epochs = 10  # Number of epochs to train for
patience = 2  # Early stopping patience

# Initialize an ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load the training and testing datasets with data augmentation and preprocessing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNet model (excluding top layer)
base_mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of the pre-trained model
x = base_mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the full model
mobilenet_model = Model(inputs=base_mobilenet.input, outputs=predictions)

# Compile the model
mobilenet_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

# Train the MobileNet-based model
mobilenet_history = mobilenet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Plot training and validation accuracy, as well as loss
plot_training_history(mobilenet_history, 'MobileNet-Based Model Training and Validation History')

# Evaluate MobileNet-based model performance
print("MobileNet-Based Model Performance:")
evaluate_model_performance(mobilenet_model, test_generator)


from keras.applications import DenseNet121

# Load the pre-trained Densenet121 model (excluding top layer)
base_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the top layers of the pre-trained model
for layer in base_densenet.layers:
    layer.trainable = False

# Add custom layers on top of the pre-trained model
x = base_densenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the full model
densenet_model = Model(inputs=base_densenet.input, outputs=predictions)

# Compile the model
densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)

# Train the Densenet121-based model
densenet_history = densenet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Plot training and validation accuracy, as well as loss
plot_training_history(densenet_history, 'Densenet121-Based Model Training and Validation History')

# Evaluate Densenet121-based model performance
print("Densenet121-Based Model Performance:")
evaluate_model_performance(densenet_model, test_generator)



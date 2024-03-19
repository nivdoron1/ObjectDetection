import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from albumentations import Compose, Resize, HorizontalFlip, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# Set the classes for your dataset
classes = ["person", "chair", "car", "dog", "bottle", "cat", "bird", "pottedplant",
           "sheep", "boat", "aeroplane", "tvmonitor", "sofa", "bicycle", "horse",
           "diningtable", "motorbike", "cow", "train", "bus"]

# Define hyperparameters
input_size = (224, 224, 3)
learning_rate = 1e-4
batch_size = 32
num_epochs = 10
num_classes = len(classes)


# Define a function to parse YOLO format labels
def parse_label_file(label_path, class_list):
    # Open the .txt file which contains the annotations
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Loop through each line
    labels = []
    for line in lines:
        # YOLO format: class x_center y_center width height
        class_id, x_center, y_center, width, height = map(float, line.split())

        # Convert YOLO annotations to bounding box coordinates
        labels.append([class_id, x_center, y_center, width, height])

    # Convert the parsed labels to a NumPy array
    return np.array(labels)


# Data augmentation and preprocessing
def augment_and_preprocess(image):
    transform = Compose([
        Resize(input_size[0], input_size[1]),
        HorizontalFlip(),
        Normalize(),
        ToTensorV2(),
    ])
    return transform(image=image)['image']

# Custom dataset loader
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels_paths, class_list, transform=None):
        self.image_paths = image_paths
        self.labels_paths = labels_paths
        self.transform = transform
        self.class_list = class_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms to the image
        if self.transform:
            image = augment_and_preprocess(image)

        # Load labels
        label = parse_label_file(self.labels_paths[index], self.class_list)

        return image, label


# Function to load the dataset paths
def load_dataset(images_dir, labels_dir):
    images_paths = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
    labels_paths = [os.path.join(labels_dir, os.path.splitext(x)[0] + '.txt') for x in os.listdir(images_dir)]

    return images_paths, labels_paths


# Load the datasets
train_images_dir = './dataset/train/images/'
train_labels_dir = './dataset/train/labels/'
valid_images_dir = './dataset/valid/images/'
valid_labels_dir = './dataset/valid/labels/'

train_image_paths, train_labels_paths = load_dataset(train_images_dir, train_labels_dir)
valid_image_paths, valid_labels_paths = load_dataset(valid_images_dir, valid_labels_dir)

# Create dataset objects
train_dataset = ImageDataset(train_image_paths, train_labels_paths, classes, transform=augment_and_preprocess)
valid_dataset = ImageDataset(valid_image_paths, valid_labels_paths, classes, transform=augment_and_preprocess)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Build the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=input_size))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of MobileNetV2
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# TensorBoard callback
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Train the model
history = model.fit(train_loader, epochs=num_epochs, validation_data=valid_loader, callbacks=[tensorboard])

# Save the model
model.save('object_detection_model.h5')

# Function to run inference on video
# def infer_on_video(video_path, model):
#    cap = cv2.VideoCapture(video_path)
#    while cap.isOpened():
#        ret, frame = cap.read()
#        if not ret:
#            break

# Preprocess the frame
# Resize the frame to match the model's expected input size
#        frame_resized = cv2.resize(frame, (input_size[0], input_size[1]))
#        frame_preprocessed = preprocess_input(frame_resized)

# Make predictions
#        predictions = model.predict(np.expand_dims(frame_preprocessed, axis=0))

# TODO: Add code to draw predictions on the frame

# Display the frame
#        cv2.imshow('Video', frame)
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            break

#    cap.release()
#    cv2.destroyAllWindows()


# Run inference on a video file
# video_file_path = 'path_to_your_video.mp4'
# infer_on_video(video_file_path, model)

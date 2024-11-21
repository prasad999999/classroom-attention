from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Conv2D, SeparableConv2D, Dropout, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Dataset path and image size
dataset_path = "C:/Users/ASUS/Desktop/Real-Time-Classroom-Monitoring-System-master/dataset/fer2013.csv"
image_size = (48, 48)

# Parameters
batch_size = 32
num_epochs = 300
input_shape = (48, 48, 1)
validation_split = 0.2
verbose = 1
num_classes = 7  # Ensure it's 7 for FER2013
patience = 50
base_path = 'C:/Users/ASUS/Desktop/Real-Time-Classroom-Monitoring-System-master/models/'  # Adjust path
l2_regularization = 0.01
model = []

# Ensure the models directory exists
if not os.path.exists(base_path):
    os.makedirs(base_path)

# Function to define the CNN model layers
def create_cnn_model(input_shape, num_classes, l2_regularization):
    regularization = l2(l2_regularization)
    img_input = Input(input_shape)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Output layer
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    # Create and compile the model
    model = Model(img_input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def plotGraph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


# Load dataset function
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in str(pixel_sequence).split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

# Preprocessing function
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Data generator for data augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Callbacks
log_file_path = os.path.join(base_path, '_emotion_training.log')
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=12, verbose=1, min_lr=1e-6)

trained_models_path = os.path.join(base_path, '_mini_XCEPTION')
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.keras'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# Load dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# Split dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

#flipped images for model2 same emotions for symmetric faces
x_train_rev = np.flip(xtrain, 2)
x_test_rev = np.flip(xtest, 2)

# Create and compile the CNN model
modelc = create_cnn_model(input_shape, num_classes, l2_regularization)

# Train the model
print("=======| Model 1 |=========")
history = modelc.fit(
    data_generator.flow(xtrain, ytrain, batch_size=batch_size),
    steps_per_epoch=len(xtrain) // batch_size if len(xtrain) // batch_size > 0 else 1,
    epochs=num_epochs,
    verbose=1,
    validation_data=(xtest, ytest),
    callbacks=callbacks
)
model.append(modelc)
plotGraph(history)




print("=======| Model 2 |=========")
modelc = create_cnn_model(input_shape, num_classes, l2_regularization)
history2 = modelc.fit(
    data_generator.flow(x_train_rev, ytrain, batch_size=batch_size),
    steps_per_epoch=len(x_train_rev) // batch_size if len(x_train_rev) // batch_size > 0 else 1,
    epochs = num_epochs,
    verbose=1,
    validation_data=(x_test_rev, ytest),
    callbacks=callbacks
)
model.append(modelc)
# plotGraph(history2)





# p_tr >> prediction on training data
# p_te >> prediction on test data

p_tr = []
p_te = []

for i, m in enumerate(model):
    if i ==0:
        p = m.predict(xtrain)
        pt = m.predict(xtest)
    else:
        p = m.predict(x_train_rev)
        pt = m.predict(x_test_rev)
    p_tr.append(p)
    p_te.append(pt)
    m.save('saved_model/cnn'+str(i)+'.h5')

print(len(model))

p_train = np.zeros((ytrain.shape[0],num_classes*len(model)))
p_test = np.zeros((ytest.shape[0],num_classes*len(model)))
for i, p in enumerate(p_tr):
    print(i)
    p_train[:,num_classes*i:num_classes*(i+1)] = p

for i, p in enumerate(p_te):
    p_test[:,num_classes*i:num_classes*(i+1)] = p

print(p_train.shape, p_test.shape)

# Trains an Conventional Neural Network on previously predicted values by the two models

batch_size = 32
num_classes = 7
epochs = 30
# Assuming num_classes and model are already defined
# Input layer
inputs = Input(shape=(num_classes * len(model),))  # Adjust input shape based on your data

# First dense layer
x = Dense(128, activation='relu')(inputs)

# Output layer
outputs = Dense(num_classes, activation='softmax')(x)

# Create model using functional API
modele = Model(inputs=inputs, outputs=outputs)

# Compile the model using Adam optimizer
modele.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),  # Using Adam optimizer
    metrics=['accuracy']
)

# Summary of the model
modele.summary()

history = modele.fit(p_train, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(p_test, ytest))

score = modele.evaluate(p_test, ytest, verbose=0)
modele.save('saved_model/ensemble.h5')

print('NN Based Ensembled Model')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
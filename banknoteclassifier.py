import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Input

dataset_path = "C:\\Users\\ERKAN EROL\\Banknote Dataset"

train = image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(128, 128),
  label_mode='categorical',
  batch_size=32)

test = image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(128, 128),
  label_mode='categorical',
  batch_size=32)


for image_batch, labels_batch in train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

for image, label in train.take(1):
    print(label)

class_names = train.class_names
print(class_names)


plt.figure(figsize=(10, 10))
for images, labels in train.take(5):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy().argmax()])
        plt.axis("off")


num_classes = len(class_names)



#Oluşturduğumuz CNN modeli 

"""
input_shape = (128, 128, 3)

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])
"""
#earlystopping 
"""
from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping('val_categorical_accuracy', patience=2, 
                    restore_best_weights=True, verbose=1)
"""

#VGG16 ile transfer learning (feature extractor olarak)

from keras.applications.vgg16 import VGG16


base_model = VGG16(include_top=False, weights='imagenet', 
                                     input_shape=(128, 128, 3))

base_model.trainable = False

inputs = Input(shape=(128, 128, 3))

x = tf.keras.applications.vgg16.preprocess_input(inputs)

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  
outputs = Dense(num_classes, activation='softmax')(x)

base_model.summary()

model = Model(inputs, outputs)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

hist = model.fit(train, epochs=50, validation_data=test )

model.summary()

acc = hist.history['categorical_accuracy']
val_acc = hist.history['val_categorical_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Categorical Accuracy')
plt.plot(val_acc, label='Validation Categorical Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Categorical Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Categorical Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
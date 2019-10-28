import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


batch_size = 32
num_classes = 70
epochs = 50
model_name = "Fruits_360.h5"
save_path = ""

path_to_train = "fruits//Training"
path_to_test = "fruits//Test"

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(path_to_train, (100, 100), batch_size=batch_size)

test_data = Generator.flow_from_directory(path_to_test, (100, 100), batch_size=batch_size)

model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (5, 5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit_generator(train_data,
                    steps_per_epoch=1000//batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_data, validation_steps = 3)

model.save(model_name)





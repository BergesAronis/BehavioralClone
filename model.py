import csv
import cv2
import numpy as np
import argparse
import sklearn
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D
from math import ceil


def get_generator(folder):
    lines = []
    with open(folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    print(len(lines))

    correction = 0.2
    def generator(samples, batch_size=32):
        num_samples = len(lines)
        while 1: # Loop forever so the generator never terminates
            shuffle(lines)
            for offset in range(0, num_samples, batch_size):
                batch_samples = lines[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(3):
                        name = folder + '/IMG/'+batch_sample[i].split('\\')[-1]
                        image = cv2.imread(name)
                        angle = float(batch_sample[3])
                        images.append(image)
                        if i == 1:
                            angle += correction
                        elif i == 2:
                            angle -= correction
                        angles.append(angle)
                        image_flipped = np.fliplr(image)
                        angle_flipped = -angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    return generator, train_samples, validation_samples

def get_model(generator, train_samples, validation_samples):

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64,3,3,activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64,3,3,activation="relu", border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images for training will be pulled from.'
    )
    parser.add_argument(
        '--update',
        action='store_true')
    args = parser.parse_args()

    generator, train_samples, validation_samples = get_generator(args.image_folder)
    print("created generator")

    if args.update:
        model = load_model(args.model)
        print("model loaded")
    else:
        model = get_model(generator, train_samples, validation_samples)
        print("model generated")


        # Set our batch size
    batch_size=32

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model.fit_generator(train_generator,
                samples_per_epoch=len(train_samples),
                validation_data=validation_generator,
                nb_val_samples=len(validation_samples),
                epochs=5, verbose=1)
    model.save(args.model)
    print("model saved as " + args.model)

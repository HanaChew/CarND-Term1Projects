import csv
from skimage import io, exposure
import cv2
import numpy as np
from sklearn.utils import shuffle
import math

samples = []
with open('./simData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#remove header line
samples = samples[1:-1]

from sklearn.model_selection import train_test_split
trainSamples, validationSamples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split('/')[-1]
                currentPath = './simData/IMG/' + filename
                image = io.imread(currentPath)
                image = image[60:135,:,:]
                image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                measurement = float(batch_sample[3])
                images.append(image)
                measurements.append(measurement)

                if measurement:
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement*-1.0)

                # left camera with auto-steer angle +0.4
                imageL = cv2.cvtColor(io.imread('simData/IMG/' + line[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                imageL = imageL[60:135,:,:]
                #scale to nVidiaSize
                imageL = cv2.resize(imageL,(200, 66), interpolation = cv2.INTER_AREA)
                images.append(imageL)
                measurements.append(min(measurement+0.2,1))
                images.append(cv2.flip(imageL,1))
                measurements.append(max(-1,measurement*-1.0-0.2))

                # right camera with auto-steer angle -0.3
                imageR = cv2.cvtColor(io.imread('simData/IMG/' + line[2].split('/')[-1]), cv2.COLOR_BGR2RGB)
                imageR = imageR[60:135,:,:]
                imageR = cv2.resize(imageR,(200, 66), interpolation = cv2.INTER_AREA)
                images.append(imageR)
                measurements.append(max(-1,measurement-0.2))
                images.append(cv2.flip(imageR,1))
                measurements.append(min(1,measurement*-1.0+0.2))

                 # equalize images
                imageEq = np.copy(image)
                for channel in range(imageEq.shape[2]):
                    imageEq[:, :, channel] = exposure.equalize_hist(imageEq[:, :, channel]) * 255
                images.append(imageEq)
                measurements.append(measurement)


            # trim image to only see section with road
            XTrain = np.array(images)
            yTrain = np.array(measurements)
            yield shuffle(XTrain, yTrain)


# Set our batch size
batchSize=32

# compile and train the model using the generator function
trainGenerator = generator(trainSamples, batch_size=batchSize)
validationGenerator = generator(validationSamples, batch_size=batchSize)

from keras.models import Sequential
from keras.layers import Lambda, Activation, Conv2D, Cropping2D, Dropout, Flatten, Dense

# create the model
model = Sequential()

model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(66,200,3)))
#model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((50,20),(0,0))))

# nVidia Model
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile the model and run it
model.compile(optimizer='adam', loss='mse')

# train the model
#model.fit(XTrain,yTrain, validation_split=0.2, shuffle=True, verbose=1)
#model.compile(loss='mse', optimizer='adam')
model.fit_generator(trainGenerator, steps_per_epoch=math.ceil(len(trainSamples)/batchSize), validation_data=validationGenerator, validation_steps=math.ceil(len(validationSamples)/batchSize), epochs=3, verbose=1)


# save the model
model.save('model.h5')

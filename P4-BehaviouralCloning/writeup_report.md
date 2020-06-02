# **Behavioral Cloning**

### Goals
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with this written report, considering the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describing how I addressed each point in my implementation.  

[//]: # (Image References)

[image1]: ./imgs/nVidia.jpg "nVidia Model"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 showing the car driving successfully driving
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
The results of this is shown in video.mp4.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. A generator (model.py lines 22-84) for efficiency and to save memory, such that batch data may be generated instead of all the image data being saved.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a model following the the model implemented by the autonomous driving team at nVidia in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

It consists of five convolutional Layers (three with a kernel size of 5x5 and 2x2 strides, and two with 3x3), followed by four fully connected layers, and the final output layer as demonstrated in the diagram below:

![nVidia Model][image1]


The data is normalized in the model using a Keras lambda layer (model.py line 100), and nonlinearities are introduced by making use of RELU layers with each of the Convolution Layers (model.py line 103-107).


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers (50%) in order to reduce overfitting (model.py line 108). And the model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 19, and 91-92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model was not tuned manually, but made use of the readily available adam optimiser, combined with a mean squared regression error (model.py line 117).

#### 4. Appropriate training data

In order to provide an appropriate amount of training data, the images were augmented and added to those provided in the dataset in a number of different ways, detailed in the next section, but the main images read before augmentation are as follows:

  1. The images and corresponding steering angle from the center camera were loaded
  2. The images from the left camera were used, adding to that a steering angle of 0.20 to correct for being too far left
  3. The imaged from the right camera were used, subtracting from that a steering angle of 0.20 to correct for being too far right

The images were also cropped to do away with unnecessary data in the top (60px) and bottom (25px) sections of images captured from the camera, and resized to be able to apply the nVidia process exactly, since that has an input of 66 x 200 x 3, as compared to our images that are 160 x 320 x 3,

---

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

My first step was to use a LeNet convolution neural network model similar to the one implemented in the previous project. However, that did not prove very successful at all, and I instead implemented the model implemented by the autonomous driving team at nVidia in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

This model architecture was developed for and tested on autonomous driving, and seemed a perfect fit for the purpose of this assignment here.

In order to prevent overfitting and to test the validity of the training results, the acquired data was split in an 80-20 ratio for training data and validation data respectively (model.py line 19). The model had a low mean-squared error for both the training and validation set, proving it was accurate, and had no overfitting.

`Final epoch training loss: 0.0204` \
`Final epoch validation loss: 0.0187`

The next step was thus to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically when the turns was too sharp and the distinction between the road and off-track was low. To improve the driving behavior in these cases, I included equalised images of the training set to see if this would make a difference. But unfortunately the results were still the same. I then proceeded to record a few more runs of the car recovering from the sharp turns in order to learn from it. And this proved to be just enough to get the car to drive autonomously around the whole track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-114) consisted of network that follows that of the nVidia architecture used for the model as follows:

![nVidia Model][image1]


As can be seen in the diagram above, this network consists of five convolutional Layers (three with a kernel size of 5x5 and 2x2 strides, and two with 3x3), followed by four fully connected layers, and the final output layer.

#### 3. Creation of the Training Set & Training Process

1. The images from the center camera were recorded, as well as the steering angle
2. The images from the left camera were used, adding to that a steering angle of 0.2 to correct for being too far left
3. The images from the right camera were used, subtracting from that a steering angle of 0.2 to correct for being too far right

To augment the above data, the images were also flipped for each of the three mounted cameras, with the corresponding negative value of steering angle.

This was still not enough, and it required an additional filter of equalised images. As well as further data collection especially on the turns.

The data is called by a random generator in batches of 64, and is split into 80% to be used for training, and 20% to be used for validation.

The first test run was always carried out with just 1 Epoch to speed things up, but once a decent model was found, I bumped it up to 5, however, realised that 3 was already sufficient as evidenced by the output below.

```sh
Epoch 1/3
236/236 [==============================] - 157s 665ms/step - loss: 0.0231 - val_loss: 0.0194
Epoch 2/3
236/236 [==============================] - 143s 605ms/step - loss: 0.0208 - val_loss: 0.0189
Epoch 3/3
236/236 [==============================] - 143s 607ms/step - loss: 0.0204 - val_loss: 0.0187
```

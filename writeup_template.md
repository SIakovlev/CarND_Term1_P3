**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/original_image.png "original_image"
[image2]: ./report_images/cropped_image.png "cropped_image"
[image3]: ./report_images/blurred_yuv_image.png "cropped_image"
[image4]: ./report_images/original_image_left.png "left_image"
[image5]: ./report_images/original_image_right.png "right_image"
[image6]: ./report_images/original_image_notflipped.png "original_image_notflipped"
[image7]: ./report_images/original_image_flipped.png "original_image_flipped"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarising the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used modified architecture from NVIDIA paper:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 68x320x3 blurred (Gaussian blur with 3x3 kernel) YUV image | 
| Lambda Layer | normalisation: image/255 - 0.5 |	
| Cropping Layer | cropping from sides by 20 pixels |	
| Convolution 1: 5x5x16    | 2x2 subsampling, RELU activation |	
| Convolution 2: 5x5x32	   | 2x2 subsampling, RELU activation |
| Convolution 3: 5x5x48    | 2x2 subsampling, RELU activation |	
| Convolution 4: 3x3x64	   | RELU activation |
| Convolution 5: 3x3x64    | RELU activation |	
| Fully connected	1: 128 outputs	| L2 weights regularisation: 0.0001 |
| Fully connected	2: 64 outputs	| L2 weights regularisation: 0.0001 |
| Fully connected	3: 16 outputs	| L2 weights regularisation: 0.0001 |
| Output Layer| 1 output value |

The idea was to get the similar number of parameters (i.e. around 350.000) and convolution layers with same filters.

#### 2. Attempts to reduce overfitting in the model

* The model contains L2 weights regularisation at each fully connected layer in order to reduce overfitting (model.py lines 199-210). 

* The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested in the following way:
  * Check model predictions for new images it has never seen before (model.py lines 228-284). To do that I wrote a separate generator (`tester`) that didn't augment imageset and just returned 2048 images at each `next(tester)` call.
  * If the result of the previous step was satisfying I run it through the simulator and ensured that the vehicle could stay on the track.

* The generator for training data augments images randomly, that helps to avoid overfitting as well.

#### 3. Model parameter tuning

The model used an adam optimizer with starting learning rate 0.001, so the learning rate was not tuned manually (model.py line 204).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I created two datasets:

* First dataset contains combination of center lane driving, recovering from the left and right sides of the road.
* Second dataset consists of turns only.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the adaptaion of NVIDIA architecture to this problem and create one large dataset with driving through the first track, record additional samples with recoveries and turns and then augment it with flips, translations and brightness changes.

The solution approach consists of several steps:

* **Neural network architecture**. My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it was used by NVIDIA for solving similar problem of behavioral cloning. 

* **Architecture tuning**. Even though I did not have problems with overfitting, after some experimentation I found that my first model had too many parameters (about 4.5M) which led to large training time. In addition I kept in mind that NVIDIA had only about 300k parameters and solved a more complicated problem, so I decided to tune the parameters of my architecture in order to reduce its capacity. 

* **Images preprocessing**. I used techniques from NVIDIA paper and adapted them for my case:
  * crop images from the bottom and top: -65 pixels off the top, -27 pixels off the bottom:
  ![alt text][image1] ![alt text][image2]
  * blur them with Gaussian filter and change their colormap to YUV:
  ![alt text][image1] ![alt text][image3]
  Once image preprocessing done for training data it should be done for simulator data as well, therefore I have added the following    changes to `drive.py` script file:
```python
...
image = Image.open(BytesIO(base64.b64decode(imgString)))
image_array = np.asarray(image)
pp_image = pre_processing(image_array)
pp_image_array = pp_image[None, :, :, :]
steering_angle = float(model.predict(pp_image_array, batch_size=1))
...
```
  The important detail here is that function `pre_processing()` converts images from RBG to YUV, whereas `opencv` read images in BGR.

* **Dataset augmentation**. In order to Dataset augmentation with flips, translations and brightness changes. To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6] ![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


The final step was to run the simulator to see how well the car was driving around track one. There were a few turns where the vehicle fell off the track. To improve the driving behavior in these cases, I recorder additional data with turns only.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

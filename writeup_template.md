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
[image8]: ./report_images/hist.png "histogramm"
[image9]: ./report_images/original_image_noshift.png "orig_noshift" 
[image10]: ./report_images/original_image_shift.png "orig_shift" 
[image11]: ./report_images/original_image_shift_crop.png "orig_shift_crop" 
[image12]: ./report_images/original_image_bright.png "bright_image" 

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

The final (model.py lines 188-202) network architecture with the following layers and layer sizes 

I used :

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

This is the modified version of NVIDIA neural network architecture. The idea was to get the similar number of parameters (i.e. around 350.000) and convolution layers with same filters.

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

The overall strategy for deriving a model architecture was to use the adaptaion of NVIDIA architecture to this problem and create one large dataset with driving through the first track, record additional samples with recoveries and turns and then augment it with flips, translations and brightness changes.

The solution approach consists of several steps:

* **Neural network architecture**. My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate for two reasons:
 * it has large number of parameters, that is important if the task is to capture driving behavior based on pictures
 * it was used by NVIDIA for solving essentialy the same problem of behavioral cloning and showed a good result

* **Architecture tuning**. Even though I did not have problems with overfitting, after some experimentation I found that my first model had too many parameters (about 4.5M) which led to large training time. In addition I kept in mind that NVIDIA had about 300k parameters and solved a more complicated problem, so I decided to tune the parameters of my architecture in order to reduce its capacity. 

* **Images preprocessing**. To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

This picture contains many elements that are not of interest, therefore I used image processing techniques from NVIDIA paper and adapted them for my case:
  * crop images from the bottom and top: -65 pixels off the top, -27 pixels off the bottom:
  ![alt text][image2]
  * blur them with Gaussian filter and change their colormap to YUV:
  ![alt text][image3]
  
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

* **Dataset augmentation**. If we train the network based on central images only, it will be biased towards straight movement since the majority of steering angles are very close to 0:
![alt text][image8]

 * In order to fix this I augmented the dataset. First, I added **corrections to steering angles corresponding to left and right images**, so that if the car sees right image its steering angle should decrease (which corresponds to car movement to the left) and steering angle should increase for left images. Also, I distinguish between two cases: straigt movement (steering angle close to 0) and turns (steering angle is around 0.2 and above). In the second case, the correction factor should be larger, which helps the car to deal with abrupt turns. This logic is shown in the code below:

```python
 # try to add robustness to turns, i.e.
 # when steering angle is large increase/decrease
 # angles corresponding to left and right images by 
 # a factor of 0.15 and take into account 30% of current angle

 if abs(angle_c) > 0.2:
     angle_l = angle_c + 0.3*abs(angle_c) + 0.15
     angle_r = angle_c - 0.3*abs(angle_c) - 0.15
 else: 
  # Not a turning case: the correction is less
     angle_l = angle_c + 0.1*abs(angle_c) + 0.03
     angle_r = angle_c - 0.1*abs(angle_c) - 0.03
```

 * Second, as can be seen from the dataset distribution it is biased towards left movement. To deal with this problem I **flipped images and reversed corresponding angles** thinking that this would help model to generalise. For example, here is an image that has then been flipped:

![alt text][image6] ![alt text][image7]

* Third, in order to improve robustness at the straight parts of the route, **random shifts** of the image to the right or left were added to the dataset. I shift the central image to the right by 10 - 20 pixels and correct the corresponding steering angle by a factor that depends on the shift. Here are some pictures that explain this idea:

![alt text][image9] 
![alt text][image10]

In order to get rid of black region in the image we crop it in the neural network layer (see the architecture) by 20 pixels from both sides and get the following result:

![alt text][image11]

* Another improvement that help to deal with sunny or dark parts of the track is brightness augmentation. I randomly change brightness of the picture by a factor of [-20, 50] and add it to the training batch. This is the examples of brightness augmentation
![alt text][image9]
![alt text][image11]

The final step was to run the simulator to see how well the car was driving around track one. There were a few turns where the vehicle fell off the track. To improve the driving behavior in these cases, I recorder additional data with turns only. After the collection process, I had about 13000 number of data points. The augmented dataset consisted of about 70000 points, therefore I used `model.fit_generator(...)` function from keras for memory savings as suggested in project materials. The ideal number of epochs was about 7-10 as evidenced by plateau of training and validation losses. I randomly shuffled the data set and put 20% of the data into a validation set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

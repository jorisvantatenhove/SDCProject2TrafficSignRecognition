# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./custom-traffic-signs-data/roundabout_sign.png "Traffic Sign 1"
[image5]: ./custom-traffic-signs-data/30_km_h_sign.png "Traffic Sign 2"
[image6]: ./custom-traffic-signs-data/turn_right_sign.png "Traffic Sign 3"
[image7]: ./custom-traffic-signs-data/double_curve.png "Traffic Sign 4"
[image8]: ./custom-traffic-signs-data/stop_sign.png "Traffic Sign 5"
[image9]: ./examples/barcharts.png "Bar Charts"

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The input data set we used had 51839 classified images of German traffic signs, divided into 43 different categories. We divided this into a trainig group of 34799 images, 4410 traffic signs were used for validation and the final 12630 images were used for testing. All images are RGB, downscaled to 32x32 pixels.

Below, we provide an exploratory visualization of the data set. For every group (so the trainig, validation and test group), we have a bar chart displaying per type what percentage of the images are classified as that type. We find that mainly speed limit signs (which are also more common than most other signs) are prevalent, together with stop, yield and priority road signs. We notice a very similar distribution in the three different groups.

![Bar charts][image9]

### Data preprocessing

An important factor that determines the performance of the neural networks is data preparation. Our input contains colored images, but all traffic signs should be identifiable without colors as well: just the shape of the sign and the picture on it should be enough information. Because of this, I decided to convert the images to grayscale

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I also normalized the image data using TensorFlow's ```per_image_standardization```.

Before applying these steps however, we first do some random data augmentation. This aims to prevent the model from overfitting on the training dataset, and becoming more robust to new inputs. We apply a random brightness modification, followed by a random contrast modification. Note that this is done before normalization, so some effects are immediately taken out again!

### Design and Test a Model Architecture

The final architecture is heavily based on LeNet, and consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x32 					|
| Flatten				| outputs 800        							|
| Fully connected		| outputs 250        							|
| RELU					|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| outputs 150        							|
| RELU					|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| outputs 80        							|
| RELU					|												|
| Fully connected		| outputs 43        							|
 
These outputs can then be converted using softmax probabilites, and compared to the one hot encoded expected output. 

To train the model, I used 10 epochs with a batch size of 128. The initial weights are generated using a truncated standard normal distribution with mean 0 and a standard deviation of 0.1. The learning rate is a static 0.001. The optimizer we used is the standard Adam optimizer.

We've chosen the LeNet model, as this is a model that is known to perform relatively well for classification problems. The two convolution layers should be able to identify the traffic signs by recognizing the line features. The 
My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.946 
* test set accuracy of 0.933

The model performs up to standard on the test set. There is some overfitting - the data augmentation combined with dropout does not appear to be strong enough for this use case, and exploring other ways to prevent overfitting could be very interesting, as we have a way better performance on the training set. 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The ```roundabout mandatory``` should be very easy to classify. The ```30 km/h``` speed limit image should also not give any problems, but it is slightly tilted. The ```turn right``` sign has some weird lighting and a small dent in the middle of the sign. The ```double curve``` is very underrepresented in the training data set, and also off center. Finally, the ```Stop``` sign is most difficult to classify, as it is covered by shadows of branches.

Here are the results of the prediction:

| Image			        |     Prediction	        					|  
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory	| Roundabout mandatory							| 
| 30 km/h     			| 30 km/h										|
| Turn right			| Turn right									|
| Double curve     		| Bumpy Road					 				|
| Stop sign				| Slippery Road      							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. We do not perform as well on these images as we do on the test set.

We found the following soft max probabilities for the 5 images:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| .99         			| Roundabout mandatory  						|
| .01     				| Speed limit (100km/h)							|
| < .01					| Priority road									|
| < .01			   		| Right-of-way at the next intersection			|
| < .01				    | End of no passing by vehicles over 3.5 metric tons   |

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| .99        			| 30 km/h	 									|
| .01     				| Stop											|
| < .01					| Roundabout mandatory							|
| < .01			   		| Right-of-way at the next intersection			| 
| < .01				    | Speed limit (50km/h)							|

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| .56         			| Turn right ahead  							|
| .43     				| Ahead only									|
| < .01					| No passing									|
| < .01			   		| Yield							 				| 
| < .01				    | Road work		      							| 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| .99         			| Children crossing								| 
| < .01    				| Go straight/left								| 
| < .01					| Right-of-way at the next intersection			| 
| < .01			   		| Dangerous curve to the right	 				| 
| < .01				    | Beware of ice/snow   							| 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| .29        			| Speed limit (30km/h)  						| 
| .24     				| Speed limit (20km/h) 							| 
| .10					| Speed limit (80km/h)							| 
| .08			   		| Speed limit (70km/h)			 				| 
| .07				    | Wild animals crossing							| 

The network was very sure about the two easiest traffic signs. The ```Turn right ahead``` apparently looks a lot like the ```Ahead only``` sign with this weird lightning. That would be very inconvenient in practice. The ```Dangerous curve``` is identified with high probability as a ```Children crossing``` sign, for some reason. The shadow covered ```Stop``` sign is understandably not identified, but the model also does not classify it as someting else - though it is pretty sure we're dealing with a speed limit here.
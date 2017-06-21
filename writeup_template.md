#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

- The cell number 2 contains basic summary of the dataset, numpy and other basic facilities are used to obtain the summary.
- Here is result of the summary, 

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```


####2. Include an exploratory visualization of the dataset.

- Cells 4 and 5 contain the bar chart visualization of the distribution of the test and training set.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?
- The image pixel values are normalized using min-max scaling to a range (-1, 1)
- Min-max scaling is simple and scales down the values of features to make the learning more robust. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					
|:---------------------:|:---------------------------------------------:| 
| Input                 |	| 32x32x3 RGB image   							
| Convolution 5x5     	|       | 1x1 stride, VALID padding, outputs 28 x 28 x 6 
| RELU			|	|												
| Dropout               |	| 0.8												
| Max pooling	      	|       | 2x2 stride, outputs 14x14x6 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Convolution 5x5     	|       | 1x1 stride, VALID padding, outputs 10 x 10 x 16
| RELU			|	|												
| Dropout               |	| 0.8												
| Max pooling	      	|       | 2x2 stride, outputs 5 x 5 x 16 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Flatten               |
| Fully connected       |	| output 400 -> 120        									
| RELU			|	|												
| Dropout               |	| 0.8												
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Fully connected       |	| output 120 -> 84 
| RELU			|	|												
| Dropout               |	| 0.8												
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Fully connected       |	| output 84 -> 43
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- The model was trained using a model which is very close to the popular LENET architecture. 
- Used Adam optimizer with the batch size of 128 and epochs of 20.
- Learning rate of 0.0008 was used to train the model.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
- Training set accuracy of 99.8%
- Validation set accuracy of 95.4% 
- Test set accuracy of 80% 

- Initially had chosen the popular `LENET` architecture.
- There was issue of overfitting, the score on the training set was close 100%, but the validation set accuracy kept dipping after specific epochs of training.
- Added dropout layers after each convolution and fully connected layers to avoid overfitting.
- Decreased learning rate to have more gradual convergence over more number of epochs.
- The most important decision was to add dropout layers to the LENET architecture to avoid overfitting.
- The validation set accuracy jumped from 70%-80% to 95% after adding dropout layers, which eliminated the issue of overfitting. 
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![caution][./test_images/caution.jpg] ![child cross][./test_images/childcross.jpg] ![Road Work][./test_images/road-work.jpg] 
![Stop][./test_images/stop.jpg] ![Yield][./test_images/yield.jpg]

Here are the reasons why there are some difficulties involved in detection of the images, 
1. The Caution board partly contains German text below it.
2. The child crossing sign board is improperly cropped.
3. Road worl images edges are not properly cropped and a tree trunk exists in the background.
4. The perspective of the image is not front on, the image is slightly slanted.
5. Contains green background of trees. 

####2. Discuss the models predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					
|:---------------------:|:---------------------------------------------:| 
| Caution			| Caution 
| Child Crossing		| Right-of-way at the next intersection 
| Road Work			| Road Work
| Stop				| Stop 
| Yield				| Yield 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 


| Probability         	|     Prediction	                |Conclusion
|:---------------------:|:--------------------------------------|--------------------------------------------------
| .99         		| Caution  				|The prediction has good confidence and its accurate.
| .92     		| Right-of-way at the next intersection |The prediction has good confidence but the prediction is wrong. 
| .99			| Road work 				|The prediction has good confidence and its accurate.
| .99			| Stop 					|The prediction has good confidence and its accurate.
| 1.0 			| Yield					|The prediction has good confidence and its accurate.					






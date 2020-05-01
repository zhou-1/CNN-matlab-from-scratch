# COVID-19 Detection on Lung Photos Based on CNN      
Team: Junwei Li, Jingyi Li, Kefan Zhang, Geng Song, Zhou Shen     
Professor: Dr. Francesco Orabona       

# Outline    
1. Why this project     
2. Our thoughts on this project    
3. Model view    
4. Data preparation    
5. Training process     
6. Results    
7. Conclusions    
8. Future Improve    


## Why this project     
Recently, the biggest and the most important thing is obviously the Coronavirus-19 in the world. Based on our team’s thoughts, two things become most important right now: 1.How to avoid getting an infection, people can do actions like WFH, wearing a mask, keeping social distance… immediately; 2.How to recognize Coronavirus cases more accurately.     
Based on the 2nd idea, our team has an ambitious goal: based on lung photos from suspected  patients, we hope our Machine Learning module can recognize whether the patient has Coronavirus or not. If this works or just helps a little bit, we think it will relieve current medical stress in hospitals.     

## Our thoughts on this project     
Why did we choose CNN as the algorithm for detection?     
Convolutional Neural Networks (CNN) are complex feed forward neural networks. CNNs are used for image classification and recognition because first, it has high accuracy. The CNN follows a hie-rar-chi-cal model which works on building a network, like a funnel, and finally gives out a fully-connected layer where all the neurons are connected to each other and the output is processed. Second, the main advantage of CNN compared to its predecessors is that it automatically detects the important features without any human supervision. For example, given many pictures of cats and dogs it learns distinctive features for each class by itself. Third, CNN is also computationally efficient. It uses special convolution and pooling operations and performs parameter sharing. This enables CNN models to run on any device, making them universally attractive.      

## Why did we use a pre-trained model?    
A pre-trained model is a model created by someone else to solve a similar problem. Instead of building a model from scratch to solve a similar one, we use the model trained on other problems as a starting point. A pre-trained model may not be 100% accurate in our application, but it saves huge efforts required to reinvent the wheel. Also, additional convolutional blocks substantially increased the training time. So we switch onto using the VGG16 pre-trained model. It has the following merits.
A pre-trained model can save huge efforts required to reinvent the wheel.   
The accuracy of the VGG16 pre-trained model is much better than MLP and CNN.     
The almost negligible time to train the dense layer with greater accuracy.      

## Model view     
VGG16 Pre-Trained Model    
VGG16 is a convolutional neural network architecture. The most unique thing about VGG16 is that instead of having a large number of hyper-parameters they focused on having convolution layers of 3x3 filters. And in the end, it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.    
Chart Flow    
Following the steps below, we adjust the parameters in the neural network.    
Feed input data into the neural network.    
The data flows from layer to layer until we have the output.     
Once we have the output, we can calculate the error    
Adjust a given parameter (weight or bias) by subtracting the derivative of the error with respect to the parameter itself.    
Iterate through the whole process.     
If we modify one layer from the network, the output of the network is going to change, which is going to change the error, which is going to change the derivative of the error with respect to the parameters. We need to be able to compute the derivatives regardless of the network architecture, regardless of the activation functions, regardless of the loss we use. We removed the last layer of VGG16, and flattened the block5-pool layer, then added our own fully connected layer to make the classification.     


## Model View Explain       
Functions we are using in our model are below:      
Activation: Tanh.m; Tanh_prime.m; relu.m; relu_prime.m; softmax.m; softmax_prime.m;  sigmoid.m; sigmoid_prime.m    
Layer: ActLayer.m; FCLayer.m; Layer.m{method: forward_propagation; backward_propagation}    
Loss: Mean_Squared_Loss.m; Mean_Squared_Loss_prime.m    
Network: Network.m{methods: add layer; use loss function; predict; fitt} Covid_model.m; test_model.m    

## Data Preparation     
Our data is collected from Kaggle, Chest X-Ray Images (Pneumonia). Since this data comes from a real clinical environment, pre-processing the raw images is absolutely necessary. In order to make use of the pretrained Vgg16 neural network, we first have to resize our raw images to three dimensional matrices with size of 224 by 224 by 3 as standard Vgg16 input. What we want from this pre trained neural network is not the final output, which is a classification label, for example, fountains, dogs, etc. What’s useful for us is the extracted features before being classified, which is the output for the last max pooling layer of this network. These features are represented with three-dimension matrices as well, with the size of 7 by 7 by 512. We also did flattening for each image feature, leading us to a feature vector with size of 1 by 25088. Same way to do with 130 images in the training set and 18 images in the test set, we have got the feature matrix and are ready for data pre-processing.    

## Split Dataset     
What also to mention is that to more convincingly improve the performance of our model, we split the given images into five subsets, 4 out of which is used for training purpose and the rest 1 is used for validation purpose.
Training process     

# Functions and different parameters we using in this project are below:
Loss function: mean square loss; Activation function: tanh; Layer : fully connected layer; Pre-trained model: vgg16; Learning rate: 0.5*[1e-1, 1e-2, 1e-3, 1e-4]; Epochs: 20/ 40.    

## Results    
Download this project and trained model and all images for training and testing this model https://github.com/lijunwei19/CNN-matlab-covid-9-x-ray     

## Conclusions      
In this project, we use VGG16 architecture as a pre-trained model for data pre-processing. Although the performance looks good, it is not really good for any medical diagnosis due to the limited dataset we can use.      

## Future Improve    
Here will be 2 future improvements for our project.      
First, the number of training images is not enough. The method to enlarge the dataset  we can think of is data Augment or collecting more x-ray scans for normal and Covid images. 
Second, try with different models to compare with our model.      



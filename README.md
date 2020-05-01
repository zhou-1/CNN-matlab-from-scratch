# CNN-matlab-from-scratch    

You can see our final report [here](https://github.com/zhou-1/CNN-matlab-from-scratch/blob/master/Report.md)      
Here is the link for our presentation: [link here](https://docs.google.com/presentation/d/1_SJeu7i0ZwzBKt9p099ARMiIrgGF_W6Ft1lAWPGFQ4s/edit?usp=sharing)     

## Why this project    
Recently, the biggest and the most important thing is obviously the Coronavirus-19 in the world. Based on our team’s thoughts, two things become most important right now: 1.How to avoid getting an infection, people can do actions like WFH, wearing a mask, keeping social distance… immediately; 2.How to recognize Coronavirus cases more accurately.     
Based on the 2nd idea, our team has an ambitious goal: based on lung photos from suspected  patients, we hope our Machine Learning module can recognize whether the patient has Coronavirus or not. If this works or just helps a little bit, we think it will relieve current medical stress in hospitals.        

##   Our thoughts on this project     
Why did we choose CNN as the algorithm for detection?      
Convolutional Neural Networks (CNN) are complex feed forward neural networks. CNNs are used for image classification and recognition because first, it has high accuracy. The CNN follows a hie-rar-chi-cal model which works on building a network, like a funnel, and finally gives out a fully-connected layer where all the neurons are connected to each other and the output is processed. Second, the main advantage of CNN compared to its predecessors is that it automatically detects the important features without any human supervision. For example, given many pictures of cats and dogs it learns distinctive features for each class by itself. Third, CNN is also computationally efficient. It uses special convolution and pooling operations and performs parameter sharing. This enables CNN models to run on any device, making them universally attractive.        

## Why did we use a pre-trained model?     
A pre-trained model is a model created by someone else to solve a similar problem. Instead of building a model from scratch to solve a similar one, we use the model trained on other problems as a starting point. A pre-trained model may not be 100% accurate in our application, but it saves huge efforts required to reinvent the wheel. Also, additional convolutional blocks substantially increased the training time. So we switch onto using the VGG16 pre-trained model. It has the following merits.     
A pre-trained model can save huge efforts required to reinvent the wheel.    
The accuracy of the VGG16 pre-trained model is much better than MLP and CNN.     
The almost negligible time to train the dense layer with greater accuracy.    
   

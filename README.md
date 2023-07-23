# ResNet - Grape disease classification

## Transfer learning
Transfer learning model means that the model was already trained on a very large and general dataset,
which should enable it to work with a large number of datasets. Training models can often be resource and time consuming,
transfer learning enables us to avoid long training times and have access to pretrained models. These models can
be easily loaded into a base model.

## ResNet
ResNet is used for image classification and is a Convolutional Neural Network (CNN) model. ResNet was originally trained
on ImageNet dataset, which has more than 14 million images. There are variations of this model, depending on the
number of the convolutional layers that are present in the model. The proposed model uses ResNet with 50 layers.

## Dataset
The [dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) is available on Kaggle. It contains
9027 images classified into four classes, all of which represent either healthy or one of 3 diseases of grape.
80% of dataset is training data, and 20% of dataset is test data. Each image has 256x256 resolution.
Every pixel value in each image is then scaled to values between 0 and 1.

## Training model
Base model uses ResNet model with 50 layers that uses ImageNet weights and is not trained further. Apart from ResNet
it also contains an output layer with 4 units, which corresponds to the number of classes. Model uses Adam
optimizer with 0.1 learning rate. It was trained over 30 epochs, but it reached the highest accuracy
on testing set after 6 epochs. The accuracies on both training and testing datasets were both over 99%.
The model is then saved as a file which can be used for further predictions without having to
train the model again.

![image](https://github.com/mato-m/resnet-grape/assets/64593617/912a0e65-a150-46b7-b88d-8e89cee51215)

## Testing model

The model was tested on 40 images downloaded from Google Images, 10 images from each class.
The model succesfully classified 38 out of 40 images (95%).

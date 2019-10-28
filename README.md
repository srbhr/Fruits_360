# Fruits_360
Fruits Detection using CNN.



Dataset used :

### Fruits 360

 A dataset of images containing fruits and vegetables

##### Dataset properties

- Total number of images: 82213.
- Training set size: 61488 images (one fruit or vegetable per image).
- Test set size: 20622 images (one fruit or vegetable per image).
- Multi-fruits set size: 103 images (more than one fruit (or fruit class) per image)
- Number of classes: 120 (fruits and vegetables).
- Image size: 100x100 pixels.

The Dataset Can be found over :  https://www.kaggle.com/moltean/fruits  and  https://github.com/Horea94/Fruit-Images-Dataset 

This is the work  of  Horea Muresan, [Mihai Oltean](https://mihaioltean.github.io/), [Fruit recognition from images using deep learning](https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning), Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

The paper introduces the dataset and an implementation of a Neural Network trained to recognized the fruits in the dataset.

## How to use this

### Requirements

The requirements.txt file, has all the packages that were in the environment at the time of training. 

* Tensorflow 2.0  (Tensorflow-GPU was used)
* Keras 2.3.1
* Matplotlib
* Numpy

### Usage 

The Images to be predicted are put under the fruits/test_images folder.

This model is pretrained with and weights is a H5py file.  Named 'Fruits_360.h5'.

The fruits.py file contains the Network Model and was used to train it. 


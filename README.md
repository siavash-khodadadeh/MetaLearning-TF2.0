# Meta Learning Framework TF 2.0

This is a framework which makes it easy to apply meta-learning techniques on different datasets.

## Requirements
Install all gpu dependencies for running TensorFlow on your system <https://www.tensorflow.org/install/gpu>.
Install all required packages with

`pip install -r requirements.txt`

## Datasets

We support a few datasets for now. We will add more datasets eventually. Please feel free to contribute by adding new 
datasets.

### Omniglot
Download [this](https://www.kaggle.com/watesoyan/omniglot/download) file and extract it into some 
directory on your system. If the link is not working (due to changes from kaggle links), try to download it from [here](https://www.kaggle.com/watesoyan/omniglot). 

Make sure to set the address of that directory the same as OMNIGLOT_RAW_DATA_ADDRESS variable in settings.py file.

### Imagenet
We download Imagenet dataset.

* Training
   * We downloaded Training images (Task 1 & 2) 138GB from the [imagenet website](http://www.image-net.org).
   * Make a directory for your raw dataset address and set the settings IMAGENET_RAW_DATA_ADDRESS variable
   to that directory address.
   * Go into that directory and extract the downloaded file into it. As a result there should be a directory 
   named 'ILSVRC2012_img_train' in your parent directory which you created.
   * In this directory, there should be 1000 tar files. In order to extract them to their own classes into the
   project data run the datasets_utils/untar_imagenet.py. As a result in your project root
   data folder there should be an imagenet folder which has all of the classes in their own directories.
   (Make sure you have enough disk space for this).
   * You do not need to download task 3 because it is a susbset of tasks 1 and 2.
* Validation
    * Download validation images (all tasks) 6.3GB from the same place that you downloaded training images (Task 1 & 2).
    * Extract it in the directory you created in previous step which should be the value of IMAGENET_RAW_DATA_ADDRESS
    variable in settings.py.
    * If you list the validation directory, you see 50,000 images in a single directory. That’s not really practical, 
    we’d like to have them in 1,000 directories as well.
    Download [this script](https://github.com/juliensimon/aws/blob/master/mxnet/imagenet/build_validation_tree.sh) and
    put it in the validation folder and run it so it will take care of the validation set.
    
* Test
    * Download test images (all tasks) 13GB from the same place that you downloaded training images (Task 1 & 2).
    * Extract it next to the train and validation folders in the same directory as you set the value of 
    IMAGENET_RAW_DATA_ADDRESS variable.
    
So after downloading and extracting and processing these files. There should be four files in your imagenet raw data
address:
 
* ILSVRC2012_img_train (which contains zip files of all the training images). You can remove this one if you need more 
space on your hard drive.
* ILSVRC2012_img_train_unzip (which contains the unzipped data for train).
* ILSVRC2012_img_val (which contains the unzipped data and for each class in its own folder).
* ILSVRC2012_img_test (which contains all the jpeg files without mentioning the class).
    
          
#### Imagenet preparation for our framework
* This part will be added soon.

### Mini-Imagenet 
You can download the dataset from the link provided at the end of 
[this github repo](https://github.com/yaoyao-liu/mini-imagenet-tools).
Go to the bottom of the readme file and choose Download tar files. 
That is the same link as [this](https://meta-transfer-learning.yaoyao-liu.com/download/).
After downloading train, val and test tar files. Extract them into a directory and set the variable
MINI_IMAGENET_RAW_DATA_ADDRESS to the address of that directory.
* Since we use [auto augment](https://tfhub.dev/google/image_augmentation/nas_cifar/1) for image augmentation on
Mini-Imagenet and the version of auto augment available on tensorflow-hub works with Tensorflow-1, we have added a 
repository for mini-imagenet which uses tensorflow1.14 and is built on top of 
[MAML code](https://github.com/cbfinn/maml). You can go to 
[this link](https://github.com/siavash-khodadadeh/maml) to use that code. 

## Citation

Please cite [this](https://arxiv.org/abs/1811.11819) article if you use this code in your work and want to publish your research.

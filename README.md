# Meta Learning Framework TF 2.0

This is a framework which makes it easy to apply meta-learning techniques on different datasets.

This repository covers a couple of meta-learning algorithms including UMTRA. For the repository of 
UMTRA during its release, please go to 
[UMTRA repo](https://github.com/siavash-khodadadeh/UMTRA-Release).

With just one-click you can read the dataset and generate train/validation/test 
classes. Then feed the tasks to meta-learning algorithm and run it and log 
train and validation metrics with tensorboard. 
Finally, you can evaluate the results on test set. 
 
We support a range of meta-learning algorithms for classification including 
[MAML](https://arxiv.org/abs/1703.03400), 
[ProtoNets](https://arxiv.org/abs/1703.05175),
[UMTRA](https://arxiv.org/abs/1811.11819), 
[LASIUM](https://arxiv.org/abs/2006.10236), 
[CACTUs](https://arxiv.org/pdf/1810.02334.pdf),
and [ANIL](https://arxiv.org/abs/1909.09157).
We are adding more and happily welcome new contributions.

For datasets, we support famous meta-learning benchmarks
including Omniglot, Mini-Imagenet, CelebA. 
We also support all datasets from [Meta-Dataset](https://arxiv.org/abs/1903.03096).
Furthermore, we support datasets for cross-domain meta-learning:
EuroSat,
PlantDisease,
ISIC,
ChestXRay8.
Last but not least, you can run algorithms on any model defined with Tensorflow 2.0 and 
Keras regardless of the architecture as far as it fits on the memory. 

Start from defining your own dataset or import one of our pre-defined datasets

`
mini_imagenet_database = MiniImagenetDatabase()
`

For MAML start 
by defining your algorithm and its hyper-parameters 
(it is a lengthy list, but you can control even the tasks for your validation set
and random seed, etc.).
```
maml = ModelAgnosticMetaLearningModel(
    database=mini_imagenet_database,
    target_database=ChestXRay8Database(),
    network_cls=MiniImagenetModel,
    n=5,
    k_ml=1,
    k_val_ml=5,
    k_val=1,
    k_val_val=15,
    k_test=15,
    k_val_test=15,
    meta_batch_size=4,
    num_steps_ml=5,
    lr_inner_ml=0.05,
    num_steps_validation=5,
    save_after_iterations=15000,
    meta_learning_rate=0.001,
    report_validation_frequency=1000,
    log_train_images_after_iteration=1000,
    num_tasks_val=100,
    clip_gradients=True,
    experiment_name='mini_imagenet',
    val_seed=42,
    val_test_batch_norm_momentum=0.0,
)
```

and then call train and evaluation by controlling how many tasks you 
want to evaluate on and do not forget the random seed for reproducibility.
```
maml.train(iterations=60000)
maml.evaluate(50, num_tasks=1000, seed=14, use_val_batch_statistics=True)
```

The code is designed in a modular way. 
For example, If you want to change the losses or try a new idea on forward loop of 
maml, you can
extend our MAML class code and update this function 
` 
def get_losses_of_tasks_batch_maml(self)
`

or if you want to just change the way to generate tasks, you can
just override function for 
data reading part:

`
def get_train_dataset(self)
`

## Requirements
Install all gpu dependencies for running TensorFlow on your system <https://www.tensorflow.org/install/gpu>.
Install all required packages with

`pip install -r requirements.txt`

## Running the code
First download dataset which you want to use as described in 
[Datasets](README.md/#Datasets) section.

For each experiment we have a python file to run it. 
For example, run the following command to run MAML on Omniglot dataset.

`python models/maml/maml_omniglot.py` 

For LASIUM experiments you can use any of these models

`python models/lasiummamlgan/mam_gan_mini_imagenet.py`

`python models/lasiummamlvae/maml_vae_omniglot.py`

`python models/lasiumprotonetsgan/protonets_vae_omniglot.py`

`python models/lasiumprotonetsvae/protonets_gan_celeba_progan.py`

Accordingly, you can run these codes on any other dataset as the 
python runner files are provided.

 
## Datasets

### Omniglot
Download [this](https://www.kaggle.com/watesoyan/omniglot/downloads/Omniglot.zip/1) file and extract it into some 
directory on your system.

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

### CelebA
Go to [this website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) in order to download CelebA dataset.
Download identity_CelebA.txt and list_attr_celeba.txt and list_eval_partitions.txt files. 
Download aligned and cropped images. The filename is Img_align_celeba.zip. 
Notes:
1. Images are first roughly aligned using similarity transformation according to the two eye locations;
2. Images are then resized to 218*178;
3. In evaluation status, "0" represents training image, "1" represents validation image, "2" represents testing image;

Put these files in a directory and extract the zip file. There should be 202,599 images in the folder.
Then set the value of CELEBA_RAW_DATA_ADDRESS variable in settings.py to the address of that folder. 
#### CelebA Task 1 - Identity recognition

#### CelebA Task 2- Attribute Assignment

### LFW
http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

The four following datasets are added for the cross-domain meta-learning challenge and 
the description to download is based on https://github.com/IBM/cdfsl-benchmark


### EuroSAT:

Home: http://madm.dfki.de/downloads

Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

Set the address of the settings variable to extracted folder.

### ISIC2018:

Home: http://challenge2018.isic-archive.com

Direct (must login): https://challenge.kitware.com/#phase/5abcbc6f56357d0139260e66

Go to direct link and download training dataset and download ground truth data
put both zip files in the same folder and unzip them.
There should be two folders in the new folder you created:
ISIC2018_Task3_Training_GroundTruth, ISIC2018_Task3_Training_Input.
Set the variable ISIC_RAW_DATASET_ADDRESS in local_settings.py for yourself.

### Plant Disease:

Home: https://www.kaggle.com/saroz014/plant-disease/

Direct: command line kaggle datasets download -d plant-disease/data

Set the setting variable to the address of the pland-disease

If you use this dataset, there is one image which is in png format and you should remove it first. Here is the address
for that:

dir: plant-disease/dataset/train/Pepper,_bell___healthy/

filename: 42f083e2-272d-4f83-ad9a-573ee90e50ec___Screen Shot 2015-05-06 at 4.01.13 PM.png


### ChestX-Ray8:

Home: https://www.kaggle.com/nih-chest-xrays/data

Direct: command line kaggle datasets download -d nih-chest-xrays/data

Download the data.zip and put it in a folder (for example names ChestX-Ray8) and unzip 
it after copying it in that folder. Then set the variable for RAW_DATASET_ADDRESS

### Meta-Dataset
Create a folder named data in project root.
This folder wil be used for fixing some minor issues in meta datasets.
For example some of the files are grayscale and they are converted to rgb and etc.


Go to https://github.com/google-research/meta-dataset

Download all the datasets based on the instructions provided in the above repository.
Do not run the converting script. This repository will convert the downloaded file
based on our need.
Then after extracting the download files set the variables to the address of 
extracted files.

* CUB_RAW_DATASEST_ADDRESS
* AIRCRAFT_RAW_DATASET_ADDRESS
* DTD_RAW_DATASET_ADDRESS
* VGG_FLOWER_RAW_DATASET_ADDRESS
* TRAFFIC_SIGN_RAW_DATASET_ADDRESS
* MSCOCO_RAW_DATASET_ADDRESS
* FUNGI_RAW_DATASET_ADDRESS

### VoxCeleb
Follow the instruction from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).

We downloaded "Audio files" and used P7Zip - Desktop from Ubuntu Software to extract the dev set ZIP file.

Extract both zip files into a new folder and set 
the variable VOXCELEB_RAW_DATASEST_ADDRESS in settings to the address of that folder.

## Citation

Please cite both  [this](https://arxiv.org/abs/1811.11819) article 
and [this](https://arxiv.org/abs/2006.10236) if you use this code in your work and want to 
publish your research.

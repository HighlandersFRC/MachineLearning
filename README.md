
# Machine Learning

This repository contains work done to develop custom convolution neural networks for object detection using Tensorflow. The focus of this repository is to reduce the barrier to entry to train large CNN, so as to make it easier to develop working networks as needed. Additionally this repository will serve as a location to document the work and sources previously used to develop the CNN's shown. For the purposes of this repository Python 3.6.9 and Tensorflow 1.14.0 were used on Ubuntu 18.04. This procedure is based upon the Tensorflow object detection system which can be found [here](https://github.com/tensorflow/models).

## Setup
### Native Installation
Before you begin - this native installation is for non-gpu based Tensorflow installations. It is recommended to use a docker container if tensorflow-gpu is desired. 

To install natively perform the following actions:



Installing TensorFlow:
~~~
pip3 install tensorflow
pip3 install pillow Cython lxml jupyter matplotlib
~~~
Installing Brew:
~~~
sudo apt-get install build-essential curl file git
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"

test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
~~~

Install Protobufs using Brew:
~~~
brew install protobuf
~~~
### Installation with Docker
Begin by downloading  the latest nvidia drivers for ubuntu and install them.  This can be done either through the command line or using the Graphical user interface. 

Next install Docker to run the tensorflow container, full instructions for installing docker can be found [here](https://docs.docker.com/install/).
Alternatively instructions for installing Docker on Ubuntu 18.04 are included below.

Add the Docker Repository:
~~~
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
~~~
Install Docker:
~~~
apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
~~~

Once Docker installation is complete validate the installation using
~~~
docker --version
~~~
Please note: Make sure your version of docker is > 19.03. Docker versions before 19.03 do not include the gpu flags needed for tensorflow gpu.

Install Docker Image for Tensorflow using:
~~~
docker pull tensorflow/tensorflow:1.14.0-gpu
~~~
Alternatively to get the latest docker image:
~~~
docker pull tensorflow/tensorflow-gpu
~~~
Run the docker image:
~~~
docker run --gpus all -it tensorflow/tensorflow:1.14.0-gpu bash
~~~
This will open the user into a docker shell which includes tensorflow.
more information about installing docker with tensorflow can be found [here](https://www.tensorflow.org/install/docker).
### 

## Configuring Models Directory
### Extracting the models tar file
Now that initial configuration has been done, the next step to training a CNN is configuring the models directory. To help simplify configuration a pre-configured models directory is included with this repository under models.tar. Download this repository and run the following:
~~~
tar -xaf models.tar
cd models
~~~
### Preparing Sample Images
Once the models directory has been downloaded the next step to building a CNN is to prepare training data. First find images pertaining to what you would like to track. This can be done either by taking pictures of the subject, or by downloading a lot of pictures online. The more pictures available to the better the classifier will be.

After acquiring pictures, move all of the pictures to the models/images directory. The pictures will then be sorted into test and train images. This can be done easily by running the rename_images script from within the models directory.
~~~
python3 rename_images.py
~~~
Now that images are sorted, label the images using the labelimg  program found [here](https://github.com/tzutalin/labelImg). It is recommended to use one of the pre-compiled executable versions. This process will need to be done for both the images/train and images/test directories independently. Make sure for each directory the the xml destination folders are set to annotations/test and annotations/train directories respectively.
### Create a label map
A label map file is used to tell the model what the labels from the xmls files mean. modify the labelmap prototype file in annotations/label_map.pbtxt and add categories for each class that you would like to train the nerual network to recognize. For example:
~~~
item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
~~~

### Building TensorFlow records files
The next step to building a CNN is building TensorFlow record files. These files contain the information needed for TensorFlow to train the model. Building the TF record files can be accomplished in two steps, first generate CSV files of the training data, then generate record files based upon the available images and CSV files. 
Install dependencies:
~~~
pip3 install pandas
~~~
Convert XML to CSV files:
~~~
python3 xml_to_csv.py -i images\train -o annotations\train_labels.csv
python3 xml_to_csv.py -i images\test -o annotations\test_labels.csv
~~~
Convert csv files to record files:
~~~
python3 generate_tfrecord.py --label=<your label from label map> --csv_input=annotations/train_labels.csv --img_path=images/train  --output_path=annotations/train.record
python generate_tfrecord.py --label=<your label from label map> --csv_input=annotations/train_labels.csv --img_path=images/train  --output_path=annotations/train.record
~~~

Once this is done make sure the records files are not empty. If the above script fails this file is often left empty and will cause the training to enter an infinite loop.

## Training the model
Before training the model there are some final steps that must be taken. 

### Compiling protobuf sources
In order to train a model, protobuf sources must be compiled This is done with the following steps:
~~~
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
~~~
Add the compiled object detection directory to your path:
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
**Important** This line of code is not persistent and will need to be re-run every time a new terminal is started. This can be fixed by adding the command to ~/.bashrc file.

### Modifying Training config file:
The next step that must be taken is modification to the training config file: ssd_mobilenet_v2_coco.config within this file there are a few key fields that need to be changed depending on your model:
* batch_size = 12 -- Increase this value until training gives Out of Memory Errors. I have found that a value of 12 works well on the TX2.
* num_classes: 1 -- Change this to be equal to the number of classes you are training for

### Training the models:
~~~
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
~~~

This process will take some time, it runs considerably faster on systems with faster GPUS.

### Exporting the Model:
The model can be exported using the following commands:
~~~
mkdir fine_tuned_model
python research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path --trained_checkpoint_prefix  train/model.ckpt-<the_highest_checkpoint_number> --output_directory fine_tuned_model
~~~
### Testing the models:
Models can be evaluated in two ways.
* Using Tensorboard 
	~~~
	$ tensorboard --logdir=training
	~~~
	This will show graphs showing the model loss and is a great way of checking to make sure a model is training properly. This does not require that the model has been exported
* Testing with the image eval script:
	~~~
	$ python3 test_model.py
	~~~	
	This will load images from the test directory and annotate and draw boxes around any detected instances of the image. The model must be exported as described above for this command to work.




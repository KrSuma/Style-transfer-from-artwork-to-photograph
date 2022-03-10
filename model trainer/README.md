_**Model Trainer to train classification using Keras and Tensorflow-gpu as a backend.**_

Credits for the trainer.py goes to:
Triage, ai@triage.com
and for the parallel.py:
kuza55


**Goal: train a default VGG19 model to be used as classifier with default weights.**

**necessary packages to be installed (using pip):**
pytest
pytest-cov
flake8
autoflake
pep8
autopep8
backports.tempfile
tensorflow-gpu
also please run 'pip install -q keras-trainer keras-model-specs stored'.

Important: Please make sure CUDA is installed for the training to be done using the tensorflow-gpu.

**Specifications:**
1. The 'data' folder is the directory where the dataset is to be copied into.
The folder will be divided into 'train' and 'val' for both training and validation datasets for the model.

2. The 'model_trainer' has the required scripts to run the TrainModel.py.

3. The 'Output' is the folder where the output file get stored. This output file, once the training is completed,\
will contain the log of the training as well as the final model and the model specification, etc.

Important to note is that the output file is stored as a .hdf5 file ( https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5) - this is the default output file for any training
done using the Keras library. The .hdf5 file can be used by the Pytorch library. A great source for a simple configuration
is instructed here on : https://www.tinymind.com/learn/terms/hdf5 .

4. The 'script' folder, where the test for GPU is located as well as our main script to specify and run the trainer.

**Instructions and Notes**

By default, the training using GPU with tensorflow-gpu as a backend for keras library is used for parallel computing - 
this speeds up the learning process by a massive amount. 

To give an example from actual test run, 23000 images for training and 2000 validation images, for classifying 2 objects 
taking 2 epochs(iterations) takes about ~2.5 hrs to train using CPU - this is achieved in only 10 minutes using a single
GPU). This is necessary to train a model from scratch using all the datasets from the Imagenet database - despite the
GPU's unparalleled speed compared to the CPU, the entire dataset in Imagenet will take roughly 5 to 7 days (based on the
test run using NDIVIA GTX 970 Graphics Card).

To test to see if the tensorflow-gpu recognizes the GPU, please run the 'GPUtest.py' and check to see if the GPU is 
recognized for training.

The default setting is as follows:
"class": "keras.applications.vgg19.VGG19",
"preprocess_func": "mean_subtraction",
"preprocess_args": [103.939, 116.779, 123.68],
"target_size": [224, 224, 3]

In the 'TrainModel.py', there is the required steps and the details in which one can specify the variables for training
the model. The training script has been commented for step by step instruction. The model can be specified 
(in our case, the vgg19 is set by default). All the rest of the specification for the CNN model later (dense, dropouts, 
weights, layers, etc.) can be specified as well - if not then the default values are used.

After the training is done, the output file will be tagged and labeled in the 'output' file directory.
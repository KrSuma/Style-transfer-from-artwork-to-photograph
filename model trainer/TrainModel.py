import os
# prepare the GPU device. 0 uses the main primary GPU.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import json
import tensorflow as tf
from keras import backend as K
from keras_trainer import Trainer
from keras_model_specs import ModelSpec
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import LearningRateScheduler
from keras import optimizers
import stored

# put the name of the train and valid folder here.
# in the case of using imagenet dataset, put the train and val folder names.
stored.sync(' YOUR TRAINING FILE ', 'data/train')
stored.sync(' YOUR VALID FILE ', 'data/valid')

# check GPU and prints out the GPU specification.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()

# we can set the dataset mean - if not set, the default values are used.
dataset_mean = [140, 120, 105]

# put the name of the model here, in our case, the vgg19. set the preprocess function and the argument for the function.
model_spec = ModelSpec.get('vgg19', preprocess_func='mean_subtraction', preprocess_args=dataset_mean)

# check the model spec and the detail.
print(json.dumps(model_spec.as_json(), indent=True))

# to train a model, need to have the data ready. the directory of these files are within the project.
train_dataset_dir = 'data/train/'
val_dataset_dir = 'data/valid/'
output_model_dir = 'output/models/'
output_logs_dir = 'output/logs/'

# Set a dropout layer with dropout rate - if not set, the default values are used.
dropout = Dropout(0.5)
# Set a dense layer with outputs - optional as well
dense = Dense(10, name='dense')
# Create a softmax activation layer - optional as well
softmax = Activation('softmax', name='softmax_activation')
# set the optimizers - optional as well
optim = optimizers.SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
# set the weights of the classes - optional as well
class_weights = {0: 13, 1: 1.4, 2: 9.8, 3: 1.8}

# creating the layers for the CNN layer
top_layers = [dropout, dense, softmax]

# freezing some layers will speed up the training but with cost to accuracy. useful for testing.
layers_to_freeze = np.arange(0,10,1)
print(layers_to_freeze)

# loading trainer
trainer = Trainer(model_spec=model_spec,
                  train_dataset_dir=train_dataset_dir,
                  val_dataset_dir=val_dataset_dir,
                  output_model_dir=output_model_dir,
                  output_logs_dir=output_logs_dir,
                  batch_size=32,
                  epochs=2,
                  workers=16,
                  max_queue_size=128,
                  num_gpus=1,
                  optimizer=optim,
                  class_weights=None,  # set to None to train from scratch!
                  verbose=False,
                  input_shape=(None, None, 3),
                  # freeze_layers_list = layers_to_freeze
                 )

# shows the model layer
trainer.model.summary()

# runs the trainer.
trainer.run()

# after the training, the summary is printed. check the output file for the final model and the details.
history = trainer.history
for i in range(0,len(history.history['val_acc'])):
    print('Epoch %d' %i)
    print('Training Accuracy was %.3f' %history.history['acc'][i])
    print('Training Loss was %.3f' %history.history['loss'][i])
    print('Validation Accuracy was %.3f' %history.history['val_acc'][i])
    print('Validation Loss was %.3f' %history.history['val_loss'][i])

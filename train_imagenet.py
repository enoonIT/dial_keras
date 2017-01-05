'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from time import time
from keras.callbacks import LearningRateScheduler
import math
from keras import backend as K
import bn_alexnet
import argparse


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: %s\n' % str(lr))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_folder")
    parser.add_argument("val_folder")
    return parser.parse_args()


def roundUp(number, multiple):
    num = number + (multiple - 1)
    return num - (num % multiple)


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("Current LR: %f" % lrate)
    return lrate


args = get_args()
batch_size = 128
v_batch_size = 128
image_size = 227
scaleFactor = 1  # 1./255
train_datagen = ImageDataGenerator(
            featurewise_center=True,
            rescale=scaleFactor,
            width_shift_range=0.15,
            height_shift_range=0.15,
            # shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=scaleFactor, featurewise_center=True)

stats_datagen = ImageDataGenerator(rescale=scaleFactor)
tmp = stats_datagen.flow_from_directory(args.val_folder,
                                        target_size=(image_size, image_size), class_mode="categorical",
                                        batch_size=500)
print("Loading samples for stats")
sample_data = tmp.next()[0]
train_datagen.fit(sample_data)
test_datagen.fit(sample_data)
print("Deleting stat samples")
del sample_data
train_generator = train_datagen.flow_from_directory(
            args.train_folder,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
            args.val_folder,
            target_size=(image_size, image_size),
            batch_size=v_batch_size,
            class_mode='categorical')

model = bn_alexnet.AlexNet(n_classes=train_generator.nb_class)
# base_model = VGG16(include_top=False, weights=None)  
# x = base_model.output
# Classification block
# x = GlobalAveragePooling2D()(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
# x = Dense(train_generator.nb_class, activation='softmax', name='predictions')(x)
# this is the model we will train
# model = Model(input=base_model.input, output=x)


lrate = LearningRateScheduler(step_decay)
checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)

sgd = SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
samples_per_epoch = roundUp(train_generator.nb_sample, batch_size)
model.fit_generator(
            train_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=60,
            validation_data=validation_generator,
            nb_val_samples=roundUp(validation_generator.nb_sample, v_batch_size),
            callbacks=[lrate, checkpointer])

model.save('final_model.h5')

'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from lmdb_iterator import AImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
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
    parser.add_argument("nb_class", type=int)
    parser.add_argument("savename", default="train")
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
batch_size = 100  # 256
v_batch_size = 50  # 100
image_size = 227
scaleFactor = 1  # 1./255
train_datagen = AImageDataGenerator(
            featurewise_center=True,
            rescale=scaleFactor,
            width_shift_range=0.0,
            height_shift_range=0.0,
            # shear_range=0.2,
            zoom_range=0.0,
            horizontal_flip=True)

test_datagen = AImageDataGenerator(rescale=scaleFactor, featurewise_center=True)

stats_datagen = AImageDataGenerator(rescale=scaleFactor)
tmp = stats_datagen.flow_from_lmdb(args.val_folder,
                                   target_size=(image_size, image_size), class_mode="categorical",
                                   batch_size=500, nb_class=args.nb_class)
print("Loading samples for stats")
sample_data = tmp.next()[0]
train_datagen.fit(sample_data)
test_datagen.fit(sample_data)
print("Deleting stat samples")
del sample_data, tmp, stats_datagen
train_generator = train_datagen.flow_from_lmdb(
            args.train_folder,
            target_size=(image_size, image_size),
            batch_size=batch_size, nb_class=args.nb_class,
            class_mode='categorical')
validation_generator = test_datagen.flow_from_lmdb(
            args.val_folder,
            target_size=(image_size, image_size),
            batch_size=v_batch_size, nb_class=args.nb_class,
            class_mode='categorical', center_crop=True)

# model = bn_alexnet.MultiBNAlexNet(n_classes=train_generator.nb_class)
model = bn_alexnet.AlexNet(n_classes=train_generator.nb_class)

lrate = LearningRateScheduler(step_decay)
savename = args.savename + "_weights.hdf5"
checkpointer = ModelCheckpoint(filepath=savename, verbose=1, save_best_only=True)

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

model.save(args.savename + '_final.h5')

import caffe
import lmdb
import PIL.Image
from StringIO import StringIO
from keras.preprocessing.image import Iterator, img_to_array, ImageDataGenerator
import numpy as np
from keras import backend as K

def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for _, value in cursor:
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        yield np.array(PIL.Image.open(s)), datum.label



class AImageDataGenerator(ImageDataGenerator):
    def flow_from_lmdb(self, lmdb_path,
                            target_size=(256, 256), color_mode='rgb',
                            nb_class=0, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',
                            follow_links=False):
        return LmdbIterator(
            lmdb_path, self,
            target_size=target_size, color_mode=color_mode,
            nb_class=nb_class, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class LmdbIterator(Iterator):

    def __init__(self, lmdb_path, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 nb_class=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.lmdb = read_lmdb(lmdb_path)
        self.nb_class = nb_class
        self.lmdb_path = lmdb_path
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        lmdb_env = lmdb.open(self.lmdb_path)
        self.nb_sample = lmdb_env.stat()["entries"]
        lmdb_env.close()
        super(LmdbIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        labels = []
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            d = next(self.lmdb, None)
            if d is None:
                self.lmdb = read_lmdb(self.lmdb_path)
                d = next(self.lmdb, None)
            (img, label)  = d
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            labels.append(label)
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            raise NotImplementedError("TODO")
        elif self.class_mode == 'binary':
            raise NotImplementedError("TODO")
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(labels):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

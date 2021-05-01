import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from utils import get_df
from dataset_utils import load_dataset


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def mask_to_image(mask):
    img = np.zeros(mask.shape[0:2] + (3,))
    # First class
    idx = mask == 1
    img[idx] = [1, 0, 0]
    # Second class
    idx = mask == 2
    img[idx] = [0, 1, 0]
    # Third class
    idx = mask == 3
    img[idx] = [0, 0, 1]
    return img


def mask_image_to_class(mask):
    classes = np.zeros((mask.shape[0], mask.shape[1]))
    idx = np.sum(np.round(mask), axis=-1) > 0
    classes[idx] = (np.argmax(mask, axis=-1) + 1)[idx]
    return classes


def get_center_point(mask):
    no_mask = False
    if len(mask[mask != 0]):
        yp, xp = np.where(mask != 0)
        x_min = np.min(xp)
        x_max = np.max(xp)
        y_min = np.min(yp)
        y_max = np.max(yp)
    else:
        x_min = 0
        x_max = mask.shape[1] - 1
        y_min = 0
        y_max = mask.shape[0] - 1
        no_mask = True

    return (x_min + x_max) / 2, (y_min + y_max) / 2, no_mask


# Data Generator class
class DataGenerator(Sequence):

    def __init__(self, data_dir='./output/', data_type='train', to_fit=True, batch_size=16,
                 slices=10, original_height=352, original_width=352, height=320, width=320, n_channels=1, n_classes=4,
                 shuffle=True, custom_keys=[],
                 input_norm_dist_percent=0.98, apply_augmentation=False, augmentation_probability=0.67):
        self.data_dir = data_dir
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (2 * slices, height, width)
        self.slices = slices
        self.height = height
        self.width = width
        self.original_height = original_height
        self.original_width = original_width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.custom_keys = custom_keys
        self.input_norm_dist_percent = input_norm_dist_percent
        self.apply_augmentation = apply_augmentation
        self.augmentation_probability = augmentation_probability
        self.augmentation_config = {
            'rotation': (0, 360),
            'scale': (0.7, 1.3),
            'alpha': (0, 350),
            'sigma': (14, 17),
        }
        self.dataset = self._get_data(data_type)
        self.epoch_count = 0
        self.on_epoch_end()

    # Returns the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    # Generates one batch of data
    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self._get_batch_indexes(index)

        # Create temp data list
        list_temp = [self.dataset[k] for k in indexes]
        x_offset = np.random.randint(-10, 10, size=len(list_temp))
        y_offset = np.random.randint(-10, 10, size=len(list_temp))

        # Generate X data
        X = self._generate_X(list_temp, x_offset, y_offset)

        # If it is training, also generates y data
        if self.to_fit:
            y = self._generate_y(list_temp, x_offset, y_offset)
            return X, y
        else:
            return X

    # Returns one batch of indexes
    def _get_batch_indexes(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return indexes

    # Update and shuffle indexes
    def on_epoch_end(self):
        self.epoch_count += 1
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # Generates a batch of X data
    def _generate_X(self, list_temp, x_offset, y_offset):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, sample in enumerate(list_temp):
            # Store sample
            data = np.zeros((*self.dim, self.n_channels))
            for j, img in enumerate(sample['data']):
                if img is not None:
                    data[j, ] = self._crop_image(img, sample['xc'] + x_offset[i], sample['yc'] + y_offset[i])
            X[i,] = self._data_augmentation(data)
        return X

    def _generate_y(self, list_temp, x_offset, y_offset):
        y = np.empty((self.batch_size, self.n_classes))
        # Generate data
        for i, sample in enumerate(list_temp):
            y[i,] = sample['label']
        return y

    def _convert_seg_image_to_one_hot_encoding(self, image):
        '''
        image must be either (x, y, z) or (x, y)
        Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
        Example (3D): if input is of shape (x, y, z), the output will ne of shape (x, y, z, n_classes)
        '''
        classes = np.arange(self.n_classes)
        out_image = np.zeros(list(image.shape) + [len(classes)], dtype=image.dtype)
        for i, c in enumerate(classes):
            x = np.zeros((len(classes)))
            x[i] = 1
            out_image[image == c] = x
        return out_image

    def _get_center_point(self, mask):
        x = list()
        y = list()

        for m in mask:
            xc, yc, no_mask = get_center_point(m)
            if not no_mask:
                x.append(xc)
                y.append(yc)

        if len(x) == 0:
            x.append(self.original_width // 2)
            y.append(self.original_height // 2)

        xc = np.round(np.mean(x)).astype(np.int)
        yc = np.round(np.mean(y)).astype(np.int)

        return xc, yc

    def _crop_images(self, data, mask, xc, yc):
        return data[:, yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2, :], mask[:, yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2]

    def _crop_image(self, data, xc, yc):
        return data[yc - self.height // 2:yc + self.height // 2, xc - self.width // 2:xc + self.width // 2, :]

    def _rotate_image(self, input_img):
        rotation = np.random.randint(self.augmentation_config['rotation'][0], self.augmentation_config['rotation'][1] + 1)
        new_input_img = np.zeros(input_img.shape)
        for i in range(input_img.shape[0]):
            new_input_img[i, ] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).rotate(rotation))
        return new_input_img

    def _scale_image(self, input_img):
        scale = np.random.random() * (self.augmentation_config['scale'][1] - self.augmentation_config['scale'][0]) + self.augmentation_config['scale'][0]
        height = int(scale * self.height)
        width = int(scale * self.width)
        new_input_img = np.zeros((input_img.shape[0], height, width, input_img.shape[3]))
        for i in range(input_img.shape[0]):
            new_input_img[i,] = np.array(Image.fromarray(input_img[i].astype(np.uint8)).resize((width, height), Image.NEAREST))
        return new_input_img

    def _elastic_transform(self, input_img):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        random_state = np.random.RandomState(None)
        alpha = np.random.randint(self.augmentation_config['alpha'][0], self.augmentation_config['alpha'][1] + 1)
        sigma = np.random.randint(self.augmentation_config['sigma'][0], self.augmentation_config['sigma'][1] + 1)

        shape = input_img.shape[1:]

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        for i in range(input_img.shape[0]):
            input_img[i] = map_coordinates(input_img[i], indices, order=1, mode='reflect').reshape(shape)

        return input_img

    def _data_augmentation(self, X):
        if self.apply_augmentation:
            X = np.concatenate([X, X, X], -1)
            X = X * 127.5 + 127.5
            if np.random.uniform() < self.augmentation_probability:
                X = self._rotate_image(X)
                X = self._scale_image(X)
                X = self._elastic_transform(X)
            X_new = self._resize_padding(2 * (np.expand_dims(rgb2gray(X), axis=-1) / 255) - 1, self.n_channels, -1, original_size=False)
            return X_new
        return X

    def _normalize_input(self, image):
        image = image - image.mean()
        pixels = image.flatten()
        delta_index = int(round(((1 - self.input_norm_dist_percent) / 2) * len(pixels)))
        pixels = np.sort(pixels)
        min = pixels[delta_index]
        max = pixels[-(delta_index + 1)]
        image = 2 * ((image - min) / (max - min)) - 1
        image[image < -1] = -1
        image[image > 1] = 1
        return image

    def _resize_padding(self, image, n_channels=None, pad_value=0, expand_dim=False, original_size=True):
        # if self.keep_z:
        slices = image.shape[0]
        # else:
        #     slices = self.slices
        if original_size:
            height = self.original_height
            width = self.original_width
        else:
            height = self.height
            width = self.width
        if n_channels is not None:
            data = np.zeros((slices, height, width, n_channels))
        else:
            data = np.zeros((slices, height, width))
        data += pad_value
        s_offest = (slices - image.shape[0]) // 2
        h_offest = (height - image.shape[1]) // 2
        w_offest = (width - image.shape[2]) // 2

        t_s_s = max(s_offest, 0)
        t_s_e = t_s_s + min(image.shape[0] + s_offest, image.shape[0]) - max(0, -s_offest)
        t_h_s = max(h_offest, 0)
        t_h_e = t_h_s + min(image.shape[1] + h_offest, image.shape[1]) - max(0, -h_offest)
        t_w_s = max(w_offest, 0)
        t_w_e = t_w_s + min(image.shape[2] + w_offest, image.shape[2]) - max(0, -w_offest)

        s_s_s = max(0, -s_offest)
        s_s_e = s_s_s + t_s_e - t_s_s
        s_h_s = max(0, -h_offest)
        s_h_e = s_h_s + t_h_e - t_h_s
        s_w_s = max(0, -w_offest)
        s_w_e = s_w_s + t_w_e - t_w_s

        if expand_dim:
            data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = np.expand_dims(image[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e], axis=-1)
        else:
            data[t_s_s:t_s_e, t_h_s:t_h_e, t_w_s:t_w_e] = image[s_s_s:s_s_e, s_h_s:s_h_e, s_w_s:s_w_e]
        return data

    def _get_data(self, type):
        df = get_df()
        dataset = []
        ds = load_dataset(root_dir=self.data_dir)
        if type == 'custom':
            keys = self.custom_keys
        else:
            keys = ds.keys()

        for key in keys:

            label = np.zeros(self.n_classes)
            label[int(df['pathology'][key - 1])] = 1

            ed_gt = self._resize_padding(ds[key]['ed_gt'])
            es_gt = self._resize_padding(ds[key]['es_gt'])
            mask = np.concatenate([ed_gt, es_gt])
            xc, yc = self._get_center_point(mask)

            if len(ds[key]['ed_data']) > self.slices:
                first = (len(ds[key]['ed_data']) - self.slices) // 2
                ed_data = self._resize_padding(self._normalize_input(ds[key]['ed_data']), self.n_channels, -1, True)
                es_data = self._resize_padding(self._normalize_input(ds[key]['es_data']), self.n_channels, -1, True)
                data = np.concatenate([ed_data[first:first + self.slices], es_data[first:first + self.slices]])
                dataset.append({
                    'data': data,
                    'label': label,
                    'xc': xc,
                    'yc': yc
                })
            else:
                first = (self.slices - len(ds[key]['ed_data'])) // 2
                ed_data = -np.ones((self.slices, self.original_height, self.original_width, self.n_channels))
                es_data = -np.ones((self.slices, self.original_height, self.original_width, self.n_channels))
                ed_data[first:first + len(ds[key]['ed_data'])] = self._resize_padding(self._normalize_input(ds[key]['ed_data']), self.n_channels, -1, True)
                es_data[first:first + len(ds[key]['es_data'])] = self._resize_padding(self._normalize_input(ds[key]['es_data']), self.n_channels, -1, True)
                data = np.concatenate([ed_data, es_data])
                dataset.append({
                    'data': data,
                    'label': label,
                    'xc': xc,
                    'yc': yc
                })

        if type == 'val':
            return dataset[int(len(dataset) * 0.8):]
        elif type == 'train':
            return dataset[:int(len(dataset) * 0.8)]
        else:
            return dataset

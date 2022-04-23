
### IMPORT LIBRARIES ###


# Standard libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

# Tensorflow/Keras:
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model

# sklearn:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# from .model import get_compiled_model
# from .data_loader import load_images,load_paths_labels, get_train_test_paths_labels, get_tf_dataset
# Normalizing output of image to scale between 0 and 1:
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# Choosing RGB channels only:
def choose_rgb_channels(image):
    return image[:, :, [2, 1, 0]]


# Loading images with normalization and RGB channels only:
def load_images(img_path, label):
    with open(img_path.decode("utf-8"), 'rb') as f:
        img = np.load(f)['x']
        img = np.moveaxis(img, 0, -1)
        img = choose_rgb_channels(img)
        img = normalize_image(img)

    return (img, label)

def load_images(img_path, label):
    with open(img_path.decode("utf-8"), 'rb') as f:
        img = np.load(f)['x']
        img = np.moveaxis(img, 0, -1)
        img = choose_rgb_channels(img)
        img = normalize_image(img)

    return (img, label)

def get_tf_dataset(paths, labels):
    tfds = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = tfds.map(lambda path, label: tf.numpy_function(load_images, [path, label], (tf.float32, tf.float32)))

    return dataset

class DataLoader():
    def __init__(self,dhs_path=None,countries_train=None, data_dir=None, min_max_labels=True):
        self.dhs_path = dhs_path
        self.df = None
        self.countries_train = countries_train
        self.data_dir = data_dir

        self.paths = None
        self.labels = None

        self.min_max_labels = min_max_labels

    # Creates image file path column out of existing dataframe variables:
    def read_dhs(self):
        self.df = pd.read_csv(self.dhs_path)

        # Asset Index is not available for all surveys. Remove rows where asset index not available
        self.df = self.df[~self.df.asset_index.isna()]

        self.df[['country', 'year', 'cluster', 'hh']] = self.df.DHSID_EA.str.split("-", expand=True)

        #Images are grouped into directories of multiple countries. Get the mapping
        country_group = pd.read_csv("data/country_group_map.csv")
        self.df = self.df.join(country_group.set_index('country'), on='country')

        # Get path (load_paths_labels)
        self.df['survey'] = self.df['DHSID_EA'].str[:10]
        self.df['cc'] = self.df['DHSID_EA'].str[:2]
        self.df['path'] = self.data_dir + os.sep + self.df.group_path + os.sep + self.df['survey'] + os.sep + self.df['DHSID_EA'] + '.npz'

    def subset_countries(self):
        self.df = self.df[self.df.country.isin(self.countries_train.split("|"))]

    def get_paths_labels(self):
        # Creating file paths and labels:
        self.paths = np.array(self.df.path.tolist())
        self.labels = np.array(self.df.asset_index.tolist()).astype(np.float32)

        # MinMax scaling of labels optional:
        if self.min_max_labels == True:
            sc = MinMaxScaler()
            self.labels = sc.fit_transform(self.labels.reshape(-1, 1))

    # Split image paths and labels into train/test based on index:
    def get_train_test_val(self):

        # Generating list of indexes for split:
        dataset_idx = np.arange(len(self.paths))

        # Splitting into train/test based on indices:
        x_train_idx, x_test_idx = train_test_split(dataset_idx, test_size=0.2, random_state=109)
        x_train_idx, x_val_idx = train_test_split(x_train_idx, test_size=0.2, random_state=109)

        # Set indexed paths and labels for each split:
        self.train_paths = self.paths[x_train_idx]
        self.train_labels = self.labels[x_train_idx]

        self.val_paths = self.paths[x_val_idx]
        self.val_labels = self.labels[x_val_idx]

        self.test_paths = self.paths[x_test_idx]
        self.test_labels = self.labels[x_test_idx]

        self.train_dataset = get_tf_dataset(self.train_paths, self.train_labels)
        self.val_dataset = get_tf_dataset(self.val_paths, self.val_labels)
        self.test_dataset = get_tf_dataset(self.test_paths, self.test_labels)



    def run(self):
        self.read_dhs()
        self.subset_countries()
        self.get_paths_labels()
        self.get_train_test_val()



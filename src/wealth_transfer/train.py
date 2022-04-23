


import tensorflow as tf

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from tensorflow.keras.applications import VGG16

import os, tempfile


# NOTE: Function comes from the following link: https://sthalles.github.io/keras-regularizer/
def add_regularization(model, regularizer=tf.keras.regularizers.l2(1e-5)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


# NOTE: Function comes from the following link: https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
def get_compiled_model(input_shape, fine_tune=0, reg_amount=None):

    # Function to compile model:
    def compile_model(model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mse'])

        return model

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(128, activation='relu')(top_model)
    top_model = Dropout(0.5)(top_model)
    output_layer = Dense(1)(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    if reg_amount != None:
        model = add_regularization(model, regularizer=tf.keras.regularizers.l2(reg_amount))

    model = compile_model(model)

    return model


class Trainer():
    def __init__(self,data_loader=None,model_hyperparams=None):
        self.data_loader = data_loader
        self.model_hyperparams = model_hyperparams

        self.n_train_imgs = len(self.data_loader.train_paths)
        self.n_val_imgs = len(self.data_loader.val_paths)

        self.model = None
        self.history = None

        self.n_channels = 3 if self.model_hyperparams.RGB_only else 8
        self.input_shape = (255, 255, self.n_channels)


    def get_model(self):
        self.model = get_compiled_model(self.input_shape, fine_tune=self.model_hyperparams.fine_tune, reg_amount=self.model_hyperparams.reg_amount)

    def train(self):
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.model_hyperparams.patience, restore_best_weights=True)
        self.history = self.model.fit(self.data_loader.train_dataset.shuffle(self.n_train_imgs).batch(self.model_hyperparams.batch_size).prefetch(self.model_hyperparams.AUTOTUNE),
                            validation_data=self.data_loader.val_dataset.shuffle(self.n_val_imgs).batch(self.model_hyperparams.batch_size).prefetch(self.model_hyperparams.AUTOTUNE),
                            epochs=self.model_hyperparams.n_epochs,
                           callbacks=[es])

    def save_model(self):
        pass

    def run(self):
        self.get_model()
        self.train()
        self.save_model()
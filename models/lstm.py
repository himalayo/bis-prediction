#!/usr/bin/env python3

import keras
import loaders.tf
import dataset_processing
import pathlib


def create_model(timepoints=180, lnode=8, fnode=16):
    cin = keras.layers.Input(batch_shape=(None, 4))
    pin = keras.layers.Input(batch_shape=(None, timepoints, 1))
    rin = keras.layers.Input(batch_shape=(None, timepoints, 1))
    cout = cin
    pout = pin
    rout = rin
    pout = keras.layers.LSTM(lnode, activation='tanh')(pout)
    rout = keras.layers.LSTM(lnode, activation='tanh')(rout)
    out = keras.layers.Concatenate()([pout, rout, cout])
    out = keras.layers.Dense(fnode, kernel_initializer=keras.initializers.HeUniform(),
                             activation='relu', kernel_regularizer=keras.regularizers.L2(1.0e-4))(out)
    out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Dense(1, kernel_initializer=keras.initializers.HeUniform(), activation='sigmoid')(out)

    model = keras.Model(inputs=[pin, rin, cin], outputs=[out])
    model.compile(loss='mae', optimizer='adam')
    return model

def train(model, train_dataset=None, validation_dataset=None,
          train_path=pathlib.Path("data/train"), validation_path=pathlib.Path("data/val"),
          checkpoint_path="weights/checkpoints/{epoch:02d}-{val_loss:.2f}.keras",
          epochs=10):

    if train_dataset is None:
        train_dataset = loaders.tf.get_dataset(train_path)

    if validation_dataset is None:
        validation_dataset = loaders.tf.get_dataset(validation_path)


    hist = model.fit(x=train_dataset, validation_data=validation_dataset, epochs=epochs,
                     callbacks=[keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, mode='auto')])

    return hist, model

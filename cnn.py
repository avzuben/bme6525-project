import numpy as np
import tensorflow as tf
from model import get_unet_3d
from data import DataGenerator

DATA_DIR = './ds-3d/'
SLICES = 10
HEIGHT = 144
WIDTH = 144
N_CHANNELS = 1
N_CLASSES = 5
EPOCHS = 50
BATCH_SIZE = 4


def create_cnn(val_set):
    weight_path = '../prob_mri_segmentation/output-3d-augmentation/' + val_set + '/best-weights'

    base_model_ed = get_unet_3d(
        input_shape=(SLICES, HEIGHT, WIDTH, N_CHANNELS),
        n_class=4
    )
    base_model_ed.load_weights(weight_path)
    for l in base_model_ed.layers:
        l.trainable = False

    base_model_es = get_unet_3d(
        input_shape=(SLICES, HEIGHT, WIDTH, N_CHANNELS),
        n_class=4
    )
    base_model_ed.load_weights(weight_path)
    for l in base_model_es.layers:
        l.trainable = False

    inputs = tf.keras.layers.Input((2 * SLICES, HEIGHT, WIDTH, N_CHANNELS))

    ed_x = inputs[:, :SLICES]
    es_x = inputs[:, SLICES:]

    ed_model = tf.keras.Model(base_model_ed.input, base_model_ed.layers[27].output)
    es_model = tf.keras.Model(base_model_es.input, base_model_es.layers[27].output)

    ed_x = ed_model(ed_x)
    es_x = es_model(es_x)

    ed_x = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 3, 3))(ed_x)
    es_x = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 3, 3))(es_x)

    ed_x = tf.keras.layers.Flatten()(ed_x)
    es_x = tf.keras.layers.Flatten()(es_x)

    x = tf.keras.layers.Concatenate(axis=1)([ed_x, es_x])
    
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)

    x = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, x)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model, base_model_ed, base_model_es


folds = []
for i in range(5):
    val_patients = np.arange(i + 1, 101, 5)
    train_patients = np.delete(np.arange(1, 101), val_patients - 1)
    folds.append((train_patients, val_patients))

cnn_predictions = np.zeros((100, 5))


for i in range(5):
    weight_path = './cnn-weights/best-weights-' + str(i)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                    filepath=weight_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='max', verbose=1)]
    model, base_model_ed, base_model_es = create_cnn(str(i + 1))
    model.summary()

    train_index, valid_index = folds[i]

    train_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                    slices=SLICES, height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                                    custom_keys=train_index, apply_augmentation=True)
    valid_generator = DataGenerator(data_type='custom', data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                                    slices=SLICES, height=HEIGHT, width=WIDTH, n_channels=N_CHANNELS, n_classes=N_CLASSES,
                                    custom_keys=valid_index)

    history = model.fit(x=train_generator, validation_data=valid_generator, epochs=EPOCHS, callbacks=callbacks)

    for l in base_model_ed.layers[18:]:
        l.trainable = True
    for l in base_model_es.layers[18:]:
        l.trainable = True

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    history_fine = model.fit(x=train_generator, validation_data=valid_generator, epochs=EPOCHS * 2, initial_epoch=history.epoch[-1] + 1, callbacks=callbacks)

    np.save('./cnn-history/cnn-history-' + str(i), history.history)
    np.save('./cnn-history/cnn-history-fine-' + str(i), history_fine.history)

    model.load_weights(weight_path)

    for k in range(valid_generator.__len__()):
        X, y = valid_generator.__getitem__(k)
        print(valid_index[k] - 1)
        cnn_predictions[valid_index[k] - 1] = model.predict(X)[0]

    np.save('./cnn_predictions', cnn_predictions)

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, normalize

from utils import get_df, get_seg_df, get_folds

EPOCHS = 100
BATCH_SIZE = 4


def create_mlp(input_shape, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


df = get_df()
one_hot = OneHotEncoder()
one_hot.fit(df[df.columns[-1:]])
y = one_hot.transform(df[df.columns[-1:]]).toarray()
X = normalize(df[df.columns[:-1]])

seg_df = get_seg_df()
X_seg = normalize(seg_df[seg_df.columns[:-1]])

folds = get_folds()

mlp_models = []
mlp_history = []
mlp_train_scores = []
mlp_valid_scores = []
mlp_predictions = np.zeros((len(y), 5))
mlp_seg_predictions = np.zeros((len(y), 5))

for i in range(len(folds)):
    train_index, valid_index = folds[i]
    weight_path = './mlp-weights/best-weights-' + str(i)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                    filepath=weight_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='max', verbose=1)]

    model = create_mlp(X.shape[-1:], y.shape[-1])
    train_ds = tf.data.Dataset.from_tensor_slices((X[train_index], y[train_index])).shuffle(len(train_index)).batch(BATCH_SIZE)
    valid_ds = tf.data.Dataset.from_tensor_slices((X[valid_index], y[valid_index])).shuffle(len(valid_index)).batch(BATCH_SIZE)
    history = model.fit(x=train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=callbacks)
    model.load_weights(weight_path)
    mlp_history.append(history)
    mlp_models.append(model)
    mlp_train_scores.append(model.evaluate(train_ds)[1])
    mlp_valid_scores.append(model.evaluate(valid_ds)[1])
    mlp_predictions[valid_index] = model(X[valid_index])
    mlp_seg_predictions[valid_index] = model(X_seg[valid_index])


print('K-Fold MLP')
print('Training accuracies: ', mlp_train_scores)
print('Validation accuracies: ', mlp_valid_scores)
print('Average training accuracy: ', np.mean(mlp_train_scores))
print('Average validation accuracy: ', np.mean(mlp_valid_scores))

np.save('mlp_predictions_prob', mlp_predictions)
np.save('mlp_seg_predictions_prob', mlp_seg_predictions)

import numpy as np
from sklearn import ensemble

from utils import get_df, get_seg_df, get_folds

df = get_df()
y = df[df.columns[-1]]
X = df[df.columns[:-1]]

seg_df = get_seg_df()
X_seg = seg_df[seg_df.columns[:-1]]

folds = get_folds()

rf_models = []
rf_train_scores = []
rf_valid_scores = []
rf_predictions = np.zeros(len(y))
rf_seg_predictions = np.zeros(len(y))

for i in range(len(folds)):
    train_index, valid_index = folds[i]
    model = ensemble.RandomForestClassifier(n_estimators=1000, bootstrap=True)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    rf_train_scores.append(model.score(X.iloc[train_index], y.iloc[train_index]))
    rf_valid_scores.append(model.score(X.iloc[valid_index], y.iloc[valid_index]))
    rf_models.append(model)
    rf_predictions[valid_index] = model.predict(X.iloc[valid_index])
    rf_seg_predictions[valid_index] = model.predict(X_seg.iloc[valid_index])

print('K-Fold Random Forest')
print('Training accuracies: ', rf_train_scores)
print('Validation accuracies: ', rf_valid_scores)
print('Average training accuracy: ', np.mean(rf_train_scores))
print('Average validation accuracy: ', np.mean(rf_valid_scores))

np.save('random_forest_predictions', rf_predictions)
np.save('random_forest_seg_predictions', rf_seg_predictions)

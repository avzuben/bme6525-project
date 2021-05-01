import numpy as np
from sklearn import svm

from utils import get_df, get_seg_df, get_folds

df = get_df()
y = df[df.columns[-1]]
X = df[df.columns[:-1]]

seg_df = get_seg_df()
X_seg = seg_df[seg_df.columns[:-1]]

folds = get_folds()

linear_train_scores = []
linear_valid_scores = []
linear_predictions = np.zeros(len(y))
linear_seg_predictions = np.zeros(len(y))
quadratic_train_scores = []
quadratic_valid_scores = []
quadratic_predictions = np.zeros(len(y))
quadratic_seg_predictions = np.zeros(len(y))
rbf_train_scores = []
rbf_valid_scores = []
rbf_predictions = np.zeros(len(y))
rbf_seg_predictions = np.zeros(len(y))

for i in range(len(folds)):
    train_index, valid_index = folds[i]
    linear_model = svm.SVC(kernel='linear')
    linear_model.fit(X.iloc[train_index], y.iloc[train_index])
    linear_train_scores.append(linear_model.score(X.iloc[train_index], y.iloc[train_index]))
    linear_valid_scores.append(linear_model.score(X.iloc[valid_index], y.iloc[valid_index]))
    linear_predictions[valid_index] = linear_model.predict(X.iloc[valid_index])
    linear_seg_predictions[valid_index] = linear_model.predict(X_seg.iloc[valid_index])

    quadratic_model = svm.SVC(kernel='poly', degree=2)
    quadratic_model.fit(X.iloc[train_index], y.iloc[train_index])
    quadratic_train_scores.append(quadratic_model.score(X.iloc[train_index], y.iloc[train_index]))
    quadratic_valid_scores.append(quadratic_model.score(X.iloc[valid_index], y.iloc[valid_index]))
    quadratic_predictions[valid_index] = quadratic_model.predict(X.iloc[valid_index])
    quadratic_seg_predictions[valid_index] = quadratic_model.predict(X_seg.iloc[valid_index])

    rbf_model = svm.SVC(kernel='rbf')
    rbf_model.fit(X.iloc[train_index], y.iloc[train_index])
    rbf_train_scores.append(rbf_model.score(X.iloc[train_index], y.iloc[train_index]))
    rbf_valid_scores.append(rbf_model.score(X.iloc[valid_index], y.iloc[valid_index]))
    rbf_predictions[valid_index] = rbf_model.predict(X.iloc[valid_index])
    rbf_seg_predictions[valid_index] = rbf_model.predict(X_seg.iloc[valid_index])

print('K-Fold Linear SVM')
print('Training accuracies: ', linear_train_scores)
print('Validation accuracies: ', linear_valid_scores)
print('Average training accuracy: ', np.mean(linear_train_scores))
print('Average validation accuracy: ', np.mean(linear_valid_scores))

print('K-Fold Quadratic SVM')
print('Training accuracies: ', quadratic_train_scores)
print('Validation accuracies: ', quadratic_valid_scores)
print('Average training accuracy: ', np.mean(quadratic_train_scores))
print('Average validation accuracy: ', np.mean(quadratic_valid_scores))

print('K-Fold RBF SVM')
print('Training accuracies: ', rbf_train_scores)
print('Validation accuracies: ', rbf_valid_scores)
print('Average training accuracy: ', np.mean(rbf_train_scores))
print('Average validation accuracy: ', np.mean(rbf_valid_scores))

np.save('linear_svm_predictions', linear_predictions)
np.save('linear_seg_svm_predictions', linear_seg_predictions)
np.save('quadratic_svm_predictions', quadratic_predictions)
np.save('quadratic_seg_svm_predictions', quadratic_seg_predictions)
np.save('rbf_svm_predictions', rbf_predictions)
np.save('rbf_seg_svm_predictions', rbf_seg_predictions)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mode
from sklearn.metrics import confusion_matrix

from utils import get_df, pathologies

ifig = 0


def plot_confusion_matrix(cm, title, class_labels=None):
    global ifig
    ifig += 1
    plt.figure(ifig)
    plt.title(title)
    sns.heatmap(cm, annot=True, linewidths=.1, fmt= '.0f', cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if class_labels is not None:
        plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels)
        plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels)
    plt.show()


df = get_df()
y = df[df.columns[-1]]

rf_pred = np.load('./random_forest_predictions.npy', allow_pickle=True)
rf_acc = np.mean(rf_pred == y)
cm = confusion_matrix(y, rf_pred)
rf_confidence = cm / np.sum(cm, 0)
rf_conf_pred = np.array([rf_confidence[:, int(pred)] for pred in rf_pred])
rf_w_pred = np.zeros((len(rf_pred), len(pathologies.keys())))
for i, pred in enumerate(rf_pred):
    rf_w_pred[i, int(pred)] = rf_acc
plot_confusion_matrix(cm, 'Random Forest - Confusion Matrix', pathologies.keys())

ls_pred = np.load('./linear_svm_predictions.npy', allow_pickle=True)
ls_acc = np.mean(ls_pred == y)
cm = confusion_matrix(y, ls_pred)
ls_confidence = cm / np.sum(cm, 0)
ls_conf_pred = np.array([ls_confidence[:, int(pred)] for pred in rf_pred])
ls_w_pred = np.zeros((len(ls_pred), len(pathologies.keys())))
for i, pred in enumerate(ls_pred):
    ls_w_pred[i, int(pred)] = ls_acc
plot_confusion_matrix(cm, 'Linear SVM - Confusion Matrix', pathologies.keys())

qs_pred = np.load('./quadratic_svm_predictions.npy', allow_pickle=True)
qs_acc = np.mean(qs_pred == y)
cm = confusion_matrix(y, qs_pred)
qs_confidence = cm / np.sum(cm, 0)
qs_conf_pred = np.array([qs_confidence[:, int(pred)] for pred in rf_pred])
qs_w_pred = np.zeros((len(qs_pred), len(pathologies.keys())))
for i, pred in enumerate(qs_pred):
    qs_w_pred[i, int(pred)] = qs_acc
plot_confusion_matrix(cm, 'Quadratic SVM - Confusion Matrix', pathologies.keys())

rs_pred = np.load('./rbf_svm_predictions.npy', allow_pickle=True)
rs_acc = np.mean(rs_pred == y)
cm = confusion_matrix(y, rs_pred)
rs_confidence = cm / np.sum(cm, 0)
rs_w_pred = np.zeros((len(rs_pred), len(pathologies.keys())))
for i, pred in enumerate(rs_pred):
    rs_w_pred[i, int(pred)] = rs_acc
rs_conf_pred = np.array([rs_confidence[:, int(pred)] for pred in rf_pred])
plot_confusion_matrix(cm, 'RBF SVM - Confusion Matrix', pathologies.keys())

nn_pred = np.load('./mlp_predictions.npy', allow_pickle=True)
nn_acc = np.mean(nn_pred == y)
cm = confusion_matrix(y, nn_pred)
nn_confidence = cm / np.sum(cm, 0)
nn_w_pred = np.zeros((len(nn_pred), len(pathologies.keys())))
for i, pred in enumerate(nn_pred):
    nn_w_pred[i, int(pred)] = nn_acc
nn_conf_pred = np.array([nn_confidence[:, int(pred)] for pred in rf_pred])
plot_confusion_matrix(cm, 'MLP - Confusion Matrix', pathologies.keys())

vote = np.array(mode(np.concatenate([[rf_pred], [ls_pred], [qs_pred], [rs_pred], [nn_pred]])))[0].flatten()
vote_acc = np.mean(vote == y)
cm = confusion_matrix(y, vote)
plot_confusion_matrix(cm, 'Vote - Confusion Matrix', pathologies.keys())

top3_vote = np.array(mode(np.concatenate([[rf_pred], [rs_pred], [nn_pred]])))[0].flatten()
top3_vote_acc = np.mean(top3_vote == y)
cm = confusion_matrix(y, top3_vote)
plot_confusion_matrix(cm, 'Top-3 Vote - Confusion Matrix', pathologies.keys())

rf_seg_pred = np.load('./random_forest_seg_predictions.npy', allow_pickle=True)
rf_seg_acc = np.mean(rf_seg_pred == y)
rf_seg_conf_pred = np.array([rf_confidence[:, int(pred)] for pred in rf_seg_pred])
rf_seg_w_pred = np.zeros((len(rf_seg_pred), len(pathologies.keys())))
for i, pred in enumerate(rf_seg_pred):
    rf_seg_w_pred[i, int(pred)] = rf_acc
cm = confusion_matrix(y, rf_seg_pred)
plot_confusion_matrix(cm, 'Random Forest - Segmentation - Confusion Matrix', pathologies.keys())

ls_seg_pred = np.load('./linear_seg_svm_predictions.npy', allow_pickle=True)
ls_seg_acc = np.mean(ls_seg_pred == y)
ls_seg_conf_pred = np.array([ls_confidence[:, int(pred)] for pred in ls_seg_pred])
ls_seg_w_pred = np.zeros((len(ls_seg_pred), len(pathologies.keys())))
for i, pred in enumerate(ls_seg_pred):
    ls_seg_w_pred[i, int(pred)] = ls_acc
cm = confusion_matrix(y, ls_seg_pred)
plot_confusion_matrix(cm, 'Linear SVM - Segmentation - Confusion Matrix', pathologies.keys())

qs_seg_pred = np.load('./quadratic_seg_svm_predictions.npy', allow_pickle=True)
qs_seg_acc = np.mean(qs_seg_pred == y)
qs_seg_conf_pred = np.array([qs_confidence[:, int(pred)] for pred in qs_seg_pred])
qs_seg_w_pred = np.zeros((len(qs_seg_pred), len(pathologies.keys())))
for i, pred in enumerate(qs_seg_pred):
    qs_seg_w_pred[i, int(pred)] = qs_acc
cm = confusion_matrix(y, qs_seg_pred)
plot_confusion_matrix(cm, 'Quadratic SVM - Segmentation - Confusion Matrix', pathologies.keys())

rs_seg_pred = np.load('./rbf_seg_svm_predictions.npy', allow_pickle=True)
rs_seg_acc = np.mean(rs_seg_pred == y)
rs_seg_conf_pred = np.array([rs_confidence[:, int(pred)] for pred in rs_seg_pred])
rs_seg_w_pred = np.zeros((len(rs_seg_pred), len(pathologies.keys())))
for i, pred in enumerate(rs_seg_pred):
    rs_seg_w_pred[i, int(pred)] = rs_acc
cm = confusion_matrix(y, rs_seg_pred)
plot_confusion_matrix(cm, 'RBF SVM - Segmentation - Confusion Matrix', pathologies.keys())

nn_seg_pred = np.load('./mlp_seg_predictions.npy', allow_pickle=True)
nn_seg_acc = np.mean(nn_seg_pred == y)
nn_seg_conf_pred = np.array([nn_confidence[:, int(pred)] for pred in nn_seg_pred])
nn_seg_w_pred = np.zeros((len(nn_seg_pred), len(pathologies.keys())))
for i, pred in enumerate(nn_seg_pred):
    nn_seg_w_pred[i, int(pred)] = nn_acc
cm = confusion_matrix(y, nn_seg_pred)
plot_confusion_matrix(cm, 'MLP - Segmentation - Confusion Matrix', pathologies.keys())

vote_seg = np.array(mode(np.concatenate([[rf_seg_pred], [ls_seg_pred], [qs_seg_pred], [rs_seg_pred], [nn_seg_pred]])))[0].flatten()
vote_seg_acc = np.mean(vote_seg == y)
cm = confusion_matrix(y, vote_seg)
plot_confusion_matrix(cm, 'Vote - Segmentation - Confusion Matrix', pathologies.keys())

top3_vote_seg = np.array(mode(np.concatenate([[rf_seg_pred], [rs_seg_pred], [nn_seg_pred]])))[0].flatten()
top3_vote_seg_acc = np.mean(top3_vote_seg == y)
cm = confusion_matrix(y, top3_vote_seg)
plot_confusion_matrix(cm, 'Top-3 Vote - Segmentation - Confusion Matrix', pathologies.keys())

cnn_pred = np.load('./cnn_predictions.npy', allow_pickle=True)
cnn_acc = np.mean(np.argmax(cnn_pred, -1) == y)
cm = confusion_matrix(y, np.argmax(cnn_pred, -1))
plot_confusion_matrix(cm, 'CNN - Confusion Matrix', pathologies.keys())

print('Accuracies')
print('Random Forest:', rf_acc)
print('Linear SVM:', ls_acc)
print('Quadratic SVM:', qs_acc)
print('RBF SVM:', rs_acc)
print('MLP:', nn_acc)
print('Vote:', vote_acc)
print('Top-3 Vote:', top3_vote_acc)

print(' ')
print('Segmentation Accuracies')
print('Random Forest:', rf_seg_acc)
print('Linear SVM:', ls_seg_acc)
print('Quadratic SVM:', qs_seg_acc)
print('RBF SVM:', rs_seg_acc)
print('MLP:', nn_seg_acc)
print('Vote:', vote_seg_acc)
print('Top-3 Vote:', top3_vote_seg_acc)

print(' ')
print('CNN Accuracies')
print('CNN:', cnn_acc)

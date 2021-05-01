import numpy as np
import matplotlib.pyplot as plt

from utils import get_df, get_seg_df

titles = {
    'lvc_edv': 'left ventricular endocardium - end-diastole volume',
    'lvm_edv': 'left ventricular epicardium - end-diastole volume',
    'rvc_edv': 'right ventricular endocardium - end-diastole volume',
    'lvc_esv': 'left ventricular endocardium - end-systole volume',
    'lvm_esv': 'left ventricular epicardium - end-systole volume',
    'rvc_esv': 'right ventricular endocardium - end-systole volume',
    'lvc_sv': 'left ventricle - stroke volume',
    'rvc_sv': 'right ventricle - stroke volume',
    'lvc_ef': 'left ventricle - ejection fraction',
    'rvc_ef': 'right ventricle - ejection fraction',
    'lvc_rvc_edr': 'left ventricle / right ventricle - end-diastole ratio',
    'lvc_rvc_esr': 'left ventricle / right ventricle - end-systole ratio',
    'lvm_lvc_edr': 'left ventricular epicardium / endocardium - end-diastole ratio',
    'lvm_lvc_esr': 'left ventricular epicardium / endocardium - end-systole ratio',
}

df = get_df()
X = df[df.columns[:-1]]

seg_df = get_seg_df()
X_seg = seg_df[seg_df.columns[:-1]]

ifig = 0

for c in df.columns[:-4]:
    min_value = min(X[c].min(), X_seg[c].min())
    max_value = min(X[c].max(), X_seg[c].max())
    diag = np.linspace(min_value, max_value, 100)
    ifig += 1
    plt.figure(ifig)
    plt.title(titles[c])
    plt.plot(diag, diag, color='black')
    plt.scatter(X[c], X_seg[c], alpha=0.6)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])
    plt.grid(which='both', axis='both')
    plt.show()

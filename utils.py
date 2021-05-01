import numpy as np
import pandas as pd

from dataset_utils import load_dataset

pathologies = {'DCM': 0, 'HCM': 1, 'MINF': 2, 'NOR': 3, 'RV': 4}


def compute_volume(seg, labels):
    mask = np.zeros_like(seg, np.bool)
    for l in labels:
        mask[seg == l] = True
    return np.sum(mask)


def get_df():

    ds = load_dataset(root_dir='./ds-3d')

    columns = ['lvc_edv', 'lvm_edv', 'rvc_edv', 'lvc_esv', 'lvm_esv', 'rvc_esv',
               'lvc_sv', 'rvc_sv', 'lvc_ef', 'rvc_ef', 'lvc_rvc_edr', 'lvc_rvc_esr',
               'lvm_lvc_edr', 'lvm_lvc_esr', 'height', 'weight', 'bmi', 'pathology']
    patients = np.zeros((len(ds.keys()), len(columns)))

    for i, pat in enumerate(ds):
        lvc_edv = compute_volume(ds[pat]['ed_gt'], [3])
        lvm_edv = compute_volume(ds[pat]['ed_gt'], [2])
        rvc_edv = compute_volume(ds[pat]['ed_gt'], [1])
        lvc_esv = compute_volume(ds[pat]['es_gt'], [3])
        lvm_esv = compute_volume(ds[pat]['es_gt'], [2])
        rvc_esv = compute_volume(ds[pat]['es_gt'], [1])
        lvc_sv = np.abs(lvc_edv - lvc_esv)
        rvc_sv = np.abs(rvc_edv - rvc_esv)
        lvc_ef = lvc_sv / lvc_edv
        rvc_ef = rvc_sv / rvc_edv
        lvc_rvc_edr = lvc_edv / rvc_edv
        lvc_rvc_esr = lvc_esv / rvc_esv
        lvm_lvc_edr = lvm_edv / lvc_edv
        lvm_lvc_esr = lvm_esv / lvc_esv
        body_mass_index = ds[pat]['weight'] / ds[pat]['height'] ** 2

        patients[i,  0] = lvc_edv
        patients[i,  1] = lvm_edv
        patients[i,  2] = rvc_edv
        patients[i,  3] = lvc_esv
        patients[i,  4] = lvm_esv
        patients[i,  5] = rvc_esv
        patients[i,  6] = lvc_sv
        patients[i,  7] = rvc_sv
        patients[i,  8] = lvc_ef
        patients[i,  9] = rvc_ef
        patients[i, 10] = lvc_rvc_edr
        patients[i, 11] = lvc_rvc_esr
        patients[i, 12] = lvm_lvc_edr
        patients[i, 13] = lvm_lvc_esr
        patients[i, 14] = ds[pat]['height']
        patients[i, 15] = ds[pat]['weight']
        patients[i, 16] = body_mass_index
        patients[i, 17] = pathologies[ds[pat]['pathology']]

    return pd.DataFrame(patients, columns=columns)


def get_seg_df():

    ds = load_dataset(root_dir='./ds-3d')

    columns = ['lvc_edv', 'lvm_edv', 'rvc_edv', 'lvc_esv', 'lvm_esv', 'rvc_esv',
               'lvc_sv', 'rvc_sv', 'lvc_ef', 'rvc_ef', 'lvc_rvc_edr', 'lvc_rvc_esr',
               'lvm_lvc_edr', 'lvm_lvc_esr', 'height', 'weight', 'bmi', 'pathology']
    patients = np.zeros((len(ds.keys()), len(columns)))

    for i, pat in enumerate(ds):

        seg = np.load('./seg-pred/' + str(pat).zfill(3) + '.npy')
        ed_seg = seg[:len(seg) // 2]
        es_seg = seg[len(seg) // 2:]

        lvc_edv = compute_volume(ed_seg, [3])
        lvm_edv = compute_volume(ed_seg, [2])
        rvc_edv = compute_volume(ed_seg, [1])
        lvc_esv = compute_volume(es_seg, [3])
        lvm_esv = compute_volume(es_seg, [2])
        rvc_esv = compute_volume(es_seg, [1])
        lvc_sv = np.abs(lvc_edv - lvc_esv)
        rvc_sv = np.abs(rvc_edv - rvc_esv)
        lvc_ef = lvc_sv / lvc_edv
        rvc_ef = rvc_sv / rvc_edv
        lvc_rvc_edr = lvc_edv / rvc_edv
        lvc_rvc_esr = lvc_esv / rvc_esv
        lvm_lvc_edr = lvm_edv / lvc_edv
        lvm_lvc_esr = lvm_esv / lvc_esv
        body_mass_index = ds[pat]['weight'] / ds[pat]['height'] ** 2

        patients[i, 0] = lvc_edv
        patients[i, 1] = lvm_edv
        patients[i, 2] = rvc_edv
        patients[i, 3] = lvc_esv
        patients[i, 4] = lvm_esv
        patients[i, 5] = rvc_esv
        patients[i, 6] = lvc_sv
        patients[i, 7] = rvc_sv
        patients[i, 8] = lvc_ef
        patients[i, 9] = rvc_ef
        patients[i, 10] = lvc_rvc_edr
        patients[i, 11] = lvc_rvc_esr
        patients[i, 12] = lvm_lvc_edr
        patients[i, 13] = lvm_lvc_esr
        patients[i, 14] = ds[pat]['height']
        patients[i, 15] = ds[pat]['weight']
        patients[i, 16] = body_mass_index
        patients[i, 17] = pathologies[ds[pat]['pathology']]

    return pd.DataFrame(patients, columns=columns)


def get_folds():
    folds = []
    for i in range(5):
        val_patients = np.arange(i + 1, 101, 5)
        train_patients = np.delete(np.arange(1, 101), val_patients - 1)
        folds.append((train_patients - 1, val_patients - 1))
    return folds

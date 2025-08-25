import os
import json
import argparse
import numpy as np
import pandas as pd
import h5py
import pickle

ALL_DISEASES_RAW = [
    "hilar congestion",
    "vascular congestion",
    "blunting of the costophrenic angle",
    "calcification",
    "pneumonia",
    "granuloma",
    "gastric distention",
    "heart failure",
    "atelectasis",
    "scoliosis",
    "pneumothorax",
    "pleural thickening",
    "infection",
    "enlargement of the cardiac silhouette",
    "hypertensive heart disease",
    "hernia",
    "tortuosity of the descending aorta ",
    "hypoxemia",
    "thymoma",
    "consolidation",
    "hematoma",
    "fracture",
    "tortuosity of the thoracic aorta ",
    "pneumomediastinum",
    "pleural effusion",
    "contusion",
    "lung opacity",
    "emphysema",
    "air collection",
    "cardiomegaly",
    "edema",
]
ALL_DISEASES = [d.strip().lower() for d in ALL_DISEASES_RAW]


def normalize_key(x):
    return (x or "").strip().lower()


def load_all_diseases_map(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {str(item["study_id"]): item for item in data}


def build_dicom_to_study_map(csv_path):
    df = pd.read_csv(csv_path)
    dicoms = df["dicom_id"].astype(str).tolist()
    studies = df["study_id"].astype(str).tolist()
    return dict(zip(dicoms, studies))


def load_sample_image_metadata(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def record_entity_keys(record):
    ent = record.get("entity", {})
    if isinstance(ent, dict):
        keys = set(ent.keys())
    elif isinstance(ent, (list, tuple, set)):
        keys = set(ent)
    elif isinstance(ent, str):
        keys = {ent}
    else:
        keys = set()
    return {normalize_key(k) for k in keys}


def build_report_disease_matrix(diseases_json, csv_path, sample_pkl, disease_vocab):
    study_map = load_all_diseases_map(diseases_json)
    dicom_to_study = build_dicom_to_study_map(csv_path)
    samples = load_sample_image_metadata(sample_pkl)
    Y = []
    for im in samples:
        dicom_id = str(im.get("image", ""))
        study_id = dicom_to_study.get(dicom_id)
        if not study_id:
            continue
        rec = study_map.get(str(study_id))
        if rec is None:
            continue
        ents = record_entity_keys(rec)
        row = [1 if d in ents else 0 for d in disease_vocab]
        Y.append(row)
    return np.asarray(Y, dtype=np.int8)


def filter_train_only(arr, splits_path, vqa_h5_path):
    splits = json.load(open(splits_path, "r"))
    with h5py.File(vqa_h5_path, "r") as hf:
        feature_idx = hf["feature_idx"][:]
    val_indices = feature_idx[np.asarray(splits["val"], dtype=np.int64)]
    test_indices = feature_idx[np.asarray(splits["test"], dtype=np.int64)]
    exclude_indices = np.unique(np.concatenate([val_indices, test_indices]))
    all_indices = np.arange(arr.shape[0], dtype=np.int64)
    include_indices = np.setdiff1d(all_indices, exclude_indices)
    return arr[include_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_root
    out_dir = args.data_root

    diseases_json = os.path.join(data_dir, "all_diseases.json")
    cxr_csv = os.path.join(data_dir, "cxr-record-list.csv")
    sample_pkl = os.path.join(data_dir, "mimic_shape_full.pkl")
    splits_path = os.path.join(data_dir, "splits_mimic_VQA.json")
    vqa_h5_path = os.path.join(data_dir, "VQA_mimic_dataset.h5")

    full_arr = build_report_disease_matrix(
        diseases_json=diseases_json,
        csv_path=cxr_csv,
        sample_pkl=sample_pkl,
        disease_vocab=ALL_DISEASES,
    )

    train_arr = filter_train_only(full_arr, splits_path, vqa_h5_path)

    out_pkl_path = os.path.join(out_dir, "train_report_disease.pkl")
    with open(out_pkl_path, "wb") as f:
        pickle.dump(train_arr.tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

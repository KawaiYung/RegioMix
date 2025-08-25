import os
import json
import argparse
import faiss
import h5py
import numpy as np


def get_nns_multi_gpu(retrieval_feat, roi_features, k=15, num_gpus=4):
    index = faiss.IndexFlatIP(retrieval_feat.shape[1])
    res = [faiss.StandardGpuResources() for _ in range(max(1, num_gpus))]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    index = faiss.index_cpu_to_gpu_multiple_py(res, index, co=co)
    index.add(retrieval_feat)
    D, I = index.search(roi_features, k)
    return index, I


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("-k", type=int, default=15)
    p.add_argument("--gpus", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    feats_h5 = os.path.join(args.data_root, "cmb_bbox_di_feats.hdf5")
    splits_json = os.path.join(args.data_root, "splits_mimic_VQA.json")
    labels_h5 = os.path.join(args.data_root, "VQA_mimic_dataset.h5")
    out_h5 = os.path.join(args.data_root, "cmb_vqadiff_nns.hdf5")

    with h5py.File(feats_h5, "r") as hf4:
        roi_features = hf4["image_features"][:]
    B, P, D = roi_features.shape
    roi_feat_d = D

    roi_features = roi_features.reshape(-1, roi_feat_d * 2).astype(np.float32)
    faiss.normalize_L2(roi_features)

    with open(splits_json, "r") as f:
        splits = json.load(f)
    with h5py.File(labels_h5, "r") as h5_label_file:
        feature_idx = h5_label_file["feature_idx"][:]

    # Exclude val and test images during rag
    val_indices = feature_idx[np.array(splits["val"], dtype=int)]
    test_indices = feature_idx[np.array(splits["test"], dtype=int)]
    exclude_indices = np.unique(np.concatenate((val_indices, test_indices)))
    all_indices = np.arange(B, dtype=int)
    include_indices = np.setdiff1d(all_indices, exclude_indices)

    db_full = roi_features.reshape(B, P, roi_feat_d * 2)
    db_full = db_full.reshape(B, P // 2, roi_feat_d * 2)
    retrieval_feat = db_full[include_indices].reshape(-1, roi_feat_d * 2).astype(np.float32)
    faiss.normalize_L2(retrieval_feat)

    _, nns = get_nns_multi_gpu(retrieval_feat, roi_features, k=args.k, num_gpus=args.gpus)
    nns = np.asarray(nns)

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as out_hf:
        out_hf.create_dataset("faiss_nns", data=nns, compression="gzip", compression_opts=4)


if __name__ == "__main__":
    main()

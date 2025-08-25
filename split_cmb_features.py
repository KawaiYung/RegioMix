import os
import json
import argparse
import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    args = p.parse_args()

    splits_path = os.path.join(args.data_root, "splits_mimic_VQA.json")
    vqa_h5_path = os.path.join(args.data_root, "VQA_mimic_dataset.h5")
    src_h5_path = os.path.join(args.data_root, "cmb_bbox_di_feats.hdf5")
    dst_h5_path = os.path.join(args.data_root, "train_cmb_bbox_di_feats.hdf5")

    splits = json.load(open(splits_path, "r"))
    with h5py.File(vqa_h5_path, "r") as h5_label_file:
        feature_idx = h5_label_file["feature_idx"][:]

    val_indices = feature_idx[splits["val"]]
    test_indices = feature_idx[splits["test"]]
    exclude_indices = np.unique(np.concatenate((val_indices, test_indices)))

    with h5py.File(src_h5_path, "r") as hf3:
        all_indices = np.arange(hf3["image_features"].shape[0])
        include_indices = np.setdiff1d(all_indices, exclude_indices)
        image_features_dataset = hf3["image_features"][include_indices]
        spatial_features_dataset = hf3["semantic_adj_matrix"][include_indices]
        image_bb_dataset = hf3["image_bb"][include_indices]
        image_adj_matrix_dataset = hf3["image_adj_matrix"][include_indices]
        bbox_label_dataset = hf3["bbox_label"][include_indices]

    with h5py.File(dst_h5_path, "w") as new_hf:
        new_hf.create_dataset("image_features", data=image_features_dataset)
        new_hf.create_dataset("semantic_adj_matrix", data=spatial_features_dataset)
        new_hf.create_dataset("image_bb", data=image_bb_dataset)
        new_hf.create_dataset("image_adj_matrix", data=image_adj_matrix_dataset)
        new_hf.create_dataset("bbox_label", data=bbox_label_dataset)


if __name__ == "__main__":
    main()

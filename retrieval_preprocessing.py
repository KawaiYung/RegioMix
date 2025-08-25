import os
import json
import numpy as np
import pickle
import h5py
import torch
import nltk
import tqdm
import argparse
import multiprocessing as mp
import gc
from dataclasses import dataclass
from typing import List, Tuple

# external
from datasets.adj import get_semantic_adj, get_adj_matrix

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)


def get_chest_xray_findings():
    return {
        "right lung": [
            "Vascular Congestion",
            "Hypoxemia",
            "Lung Opacity",
            "Air Collection",
            "Edema",
            "Infection",
            "Heart Failure",
            "Pneumonia",
            "Consolidation",
            "Atelectasis",
            "Granuloma",
            "Emphysema",
            "Pleural Effusion",
            "Fracture",
        ],
        "right upper lung zone": ["Pneumothorax"],
        "right mid lung zone": ["Pneumothorax", "Pleural Effusion"],
        "right lower lung zone": ["Pleural Effusion"],
        "right hilar structures": ["Hilar Congestion"],
        "right apical zone": ["Pneumothorax"],
        "right costophrenic angle": ["Blunting of the Costophrenic Angle", "Pleural Effusion"],
        "right hemidiaphragm": ["Gastric Distention", "Pleural Thickening", "Hernia"],
        "left lung": [
            "Vascular Congestion",
            "Hypoxemia",
            "Lung Opacity",
            "Air Collection",
            "Edema",
            "Infection",
            "Heart Failure",
            "Pneumonia",
            "Consolidation",
            "Atelectasis",
            "Granuloma",
            "Emphysema",
            "Hematoma",
            "Fracture",
        ],
        "left upper lung zone": ["Pneumothorax"],
        "left mid lung zone": ["Pneumothorax", "Pleural Effusion"],
        "left lower lung zone": ["Pleural Effusion"],
        "left hilar structures": ["Hilar Congestion"],
        "left apical zone": ["Pneumothorax"],
        "left costophrenic angle": ["Blunting of the Costophrenic Angle", "Pleural Effusion"],
        "left hemidiaphragm": ["Gastric Distention", "Pleural Thickening", "Hernia"],
        "trachea": ["Infection", "scoliosis"],
        "right clavicle": ["Hematoma", "Fracture"],
        "left clavicle": ["Hematoma", "Fracture"],
        "aortic arch": ["Calcification", "Tortuosity of the Descending Aorta", "Tortuosity of the Thoracic Aorta"],
        "upper mediastinum": [
            "Calcification",
            "Thymoma",
            "scoliosis",
            "Tortuosity of the Descending Aorta",
            "Tortuosity of the Thoracic Aorta",
            "Vascular Congestion",
            "pneumomediastinum",
        ],
        "svc": ["Vascular Congestion"],
        "cardiac silhouette": [
            "Heart Failure",
            "Enlargement of the Cardiac Silhouette",
            "Hypertensive Heart Disease",
            "Cardiomegaly",
            "scoliosis",
        ],
        "cavoatrial junction": ["Hypertensive Heart Disease"],
        "right atrium": ["Heart Failure"],
        "carina": ["Vascular Congestion"],
    }


def get_disease_loc_dict():
    ans_loc_dict = {}
    ans_loc_dict["hilar congestion"] = ["right hilar structures", "left hilar structures"]
    ans_loc_dict["vascular congestion"] = ["right lung", "left lung"]
    ans_loc_dict["blunting of the costophrenic angle"] = ["right costophrenic angle", "left costophrenic angle"]
    ans_loc_dict["calcification"] = ["aortic arch", "upper mediastinum"]
    ans_loc_dict["pneumonia"] = ["right lung", "left lung"]
    ans_loc_dict["granuloma"] = ["right lung", "left lung"]
    ans_loc_dict["gastric distention"] = ["left lower lung zone"]
    ans_loc_dict["heart failure"] = ["cardiac silhouette", "right lung", "left lung"]
    ans_loc_dict["atelectasis"] = ["right lower lung zone", "left lower lung zone"]
    ans_loc_dict["scoliosis"] = ["spine"]
    ans_loc_dict["pneumothorax"] = ["right upper lung zone", "left upper lung zone"]
    ans_loc_dict["pleural thickening"] = ["right lower lung zone", "left lower lung zone"]
    ans_loc_dict["infection"] = ["right lung", "left lung"]
    ans_loc_dict["enlargement of the cardiac silhouette"] = ["cardiac silhouette"]
    ans_loc_dict["hypertensive heart disease"] = ["cardiac silhouette"]
    ans_loc_dict["hernia"] = ["diaphragm regions"]
    ans_loc_dict["tortuosity of the descending aorta"] = ["aortic arch"]
    ans_loc_dict["hypoxemia"] = ["right lung", "left lung"]
    ans_loc_dict["thymoma"] = ["upper mediastinum"]
    ans_loc_dict["consolidation"] = ["right lung", "left lung"]
    ans_loc_dict["hematoma"] = ["other"]
    ans_loc_dict["fracture"] = ["right clavicle", "left clavicle", "right lung", "left lung"]
    ans_loc_dict["tortuosity of the thoracic aorta"] = ["aortic arch"]
    ans_loc_dict["pneumomediastinum"] = ["mediastinum"]
    ans_loc_dict["pleural effusion"] = ["right costophrenic angle", "left costophrenic angle"]
    ans_loc_dict["contusion"] = ["other"]
    ans_loc_dict["lung opacity"] = ["right lung", "left lung"]
    ans_loc_dict["emphysema"] = ["right upper lung zone", "left upper lung zone"]
    ans_loc_dict["air collection"] = ["right lung", "left lung"]
    ans_loc_dict["cardiomegaly"] = ["cardiac silhouette"]
    ans_loc_dict["edema"] = ["right lung", "left lung", "edema"]
    return ans_loc_dict


all_diseases = [
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


def map_findings_to_indices(chest_xray_findings, findings_list):
    findings_list_lower = [f.lower() for f in findings_list]
    result = []
    for key in chest_xray_findings:
        current_key_indices = []
        for finding in chest_xray_findings[key]:
            fl = finding.lower()
            if fl in findings_list_lower:
                current_key_indices.append(findings_list_lower.index(fl))
        result.append(current_key_indices)
    return result


def get_caption(adding, dropping):
    adding = list(adding)
    dropping = list(dropping)
    if len(adding) == 1:
        out1 = "the main image has an additional finding of"
    elif len(adding) > 1:
        out1 = "the main image has additional findings of"
    else:
        out1 = ""
    for item in adding:
        out1 += (" " + item) if len(adding) == 1 or item == adding[-1] else (" " + item + ",")
    if len(adding) != 0:
        out1 += " than the reference image. "
    if len(dropping) == 1:
        out2 = "the main image is missing the finding of"
    elif len(dropping) > 1:
        out2 = "the main image is missing the findings of"
    else:
        out2 = ""
    for item in dropping:
        out2 += (" " + item) if len(dropping) == 1 or item == dropping[-1] else (" " + item + ",")
    if len(dropping) != 0:
        out2 += " than the reference image. "
    return out1 + out2


def get_label(ids, max_seq):
    out = np.zeros(max_seq, dtype=np.int64)
    n = min(len(ids), max_seq)
    out[:n] = np.asarray(ids[:n], dtype=np.int64)
    return out


def map_labels_to_diseases(label, label_to_related_disease):
    # label: (26, k, 1) or (26, k)
    label = np.asarray(label)
    disease_array = np.empty(len(label_to_related_disease), dtype=object)
    for i, v in enumerate(label_to_related_disease):
        disease_array[i] = v
    mapped = disease_array[label.squeeze()]
    return mapped.reshape(label.shape[0], label.shape[1], -1)


def get_diseases_summary(result_array, int_mapped):
    processed = np.full(result_array.shape, -1)
    for row_idx, (indices_row, result_row) in enumerate(zip(int_mapped, result_array)):
        indices = np.array(indices_row[0], dtype=int)
        processed[row_idx, indices] = [result_row[i] for i in indices]
    return processed


def get_diseases_average(array):
    nonneg = (array >= 0).astype(np.int64)
    s = np.sum(np.where(array >= 0, array, 0), axis=0)
    c = np.sum(nonneg, axis=0)
    c[c == 0] = 1
    avg = s / c
    avg[c == 1] = 0
    return avg


def map_indices(shape, indices):
    indices = np.asarray(indices)
    flat = indices.reshape(-1)
    return np.stack((flat // shape, flat % shape), axis=-1).reshape(*indices.shape, 2)


def save_h5(
    filename: str,
    reaug_features,
    rag_label,
    diseases_avg,
    reaug_state_list,
    rag_bb,
    rag_semantic_adj,
    rag_adj_matrix,
    test_num_sources_per_image: int,
    times: int,
    length: int,
    k: int,
):

    label_comb = int(test_num_sources_per_image * k // 2)
    mode = "w" if times == 0 else "a"
    with h5py.File(filename, mode) as h5f:
        if times == 0:
            h5f.create_dataset(
                "reaug_features",
                (length, 2, test_num_sources_per_image, k, 1024),
                maxshape=(None, 2, test_num_sources_per_image, k, 1024),
                chunks=(100, 2, test_num_sources_per_image, k, 1024),
                dtype="float32",
            )
            h5f.create_dataset(
                "rag_label",
                (length, 2, label_comb),
                maxshape=(None, 2, label_comb),
                chunks=(100, 2, label_comb),
                dtype="int64",
            )
            h5f.create_dataset(
                "diseases_avg",
                (length, 2, 31),
                maxshape=(None, 2, 31),
                chunks=(100, 2, 31),
                dtype="int64",
            )
            h5f.create_dataset(
                "reaug_state_list",
                (length, 90),
                maxshape=(None, 90),
                chunks=(100, 90),
                dtype="int64",
            )
            h5f.create_dataset(
                "rag_bb",
                (length, 2, k, test_num_sources_per_image, 4),
                maxshape=(None, 2, k, test_num_sources_per_image, 4),
                chunks=(100, 2, k, test_num_sources_per_image, 4),
                dtype="float64",
            )
            h5f.create_dataset(
                "rag_semantic_adj",
                (length, 2, k, 100, 100),
                maxshape=(None, 2, k, 100, 100),
                chunks=(100, 2, k, 100, 100),
                dtype="int64",
            )
            h5f.create_dataset(
                "rag_adj_matrix",
                (length, 2, k, 100, 100),
                maxshape=(None, 2, k, 100, 100),
                chunks=(100, 2, k, 100, 100),
                dtype="int64",
            )

        # resize & write
        def rz(name, new_len):
            h5f[name].resize([new_len, *h5f[name].shape[1:]])

        start = times * length
        end = start + length

        rz("reaug_features", end)
        h5f["reaug_features"][start:end] = reaug_features
        rz("rag_label", end)
        h5f["rag_label"][start:end] = rag_label
        rz("diseases_avg", end)
        h5f["diseases_avg"][start:end] = diseases_avg
        rz("reaug_state_list", end)
        h5f["reaug_state_list"][start:end] = reaug_state_list
        rz("rag_bb", end)
        h5f["rag_bb"][start:end] = rag_bb
        rz("rag_semantic_adj", end)
        h5f["rag_semantic_adj"][start:end] = rag_semantic_adj
        rz("rag_adj_matrix", end)
        h5f["rag_adj_matrix"][start:end] = rag_adj_matrix

        print("reaug_features_dataset.shape", h5f["reaug_features"].shape)


@dataclass
class Context:
    faiss_nns: np.ndarray
    feature_idx: np.ndarray
    faiss_features: np.ndarray
    faiss_label: np.ndarray
    faiss_bb: np.ndarray
    ana_feat: np.ndarray
    ana_label: np.ndarray
    ana_bb: np.ndarray
    report_disease: np.ndarray
    word_to_idx: dict
    label_to_related_disease: List[List[int]]
    k: int


def process_chunk(ctx: Context, split_idxs_local: List[int], feature_extraction_path: str) -> Tuple:
    rag_features_list = []
    rag_label_list = []
    diseases_avg_list = []
    reaug_state_list_list = []
    rag_bb_list = []
    rag_semantic_adj_list = []
    rag_adj_matrix_list = []

    for idx in tqdm.tqdm(range(len(split_idxs_local))):
        d_reaug_idx = ctx.faiss_nns[ctx.feature_idx[idx, 0], :, : ctx.k]
        q_reaug_idx = ctx.faiss_nns[ctx.feature_idx[idx, 1], :, : ctx.k]

        d_reaug_bbox = ctx.faiss_features[d_reaug_idx, :]
        q_reaug_bbox = ctx.faiss_features[q_reaug_idx, :]
        d_reaug_ana = ctx.ana_feat[d_reaug_idx, :]
        q_reaug_ana = ctx.ana_feat[q_reaug_idx, :]
        d_reaug_features = np.concatenate((d_reaug_ana, d_reaug_bbox), axis=0)
        q_reaug_features = np.concatenate((q_reaug_ana, q_reaug_bbox), axis=0)

        d_reaug_bb_bbox = torch.from_numpy(ctx.faiss_bb[d_reaug_idx, :]).double()
        q_reaug_bb_bbox = torch.from_numpy(ctx.faiss_bb[q_reaug_idx, :]).double()
        d_reaug_bb_ana = torch.from_numpy(ctx.ana_bb[d_reaug_idx, :]).double()
        q_reaug_bb_ana = torch.from_numpy(ctx.ana_bb[q_reaug_idx, :]).double()
        d_reaug_bb = np.concatenate((d_reaug_bb_ana, d_reaug_bb_bbox), axis=0)
        q_reaug_bb = np.concatenate((q_reaug_bb_ana, q_reaug_bb_bbox), axis=0)

        P, _, _ = d_reaug_bb.shape
        d_reaug_bb = np.transpose(d_reaug_bb, (1, 0, 2))
        q_reaug_bb = np.transpose(q_reaug_bb, (1, 0, 2))

        rag_d_adj_matrix = get_adj_matrix(d_reaug_bb)
        rag_q_adj_matrix = get_adj_matrix(q_reaug_bb)

        d_label_reaug = ctx.ana_label[d_reaug_idx]
        q_label_reaug = ctx.ana_label[q_reaug_idx]
        d_bb_label_reaug = ctx.faiss_label[d_reaug_idx]
        q_bb_label_reaug = ctx.faiss_label[q_reaug_idx]

        rag_d_semantic_adj = get_semantic_adj(
            np.transpose(d_label_reaug, (1, 0)), np.transpose(d_bb_label_reaug, (1, 0)), feature_extraction_path
        )
        rag_q_semantic_adj = get_semantic_adj(
            np.transpose(q_label_reaug, (1, 0)), np.transpose(q_bb_label_reaug, (1, 0)), feature_extraction_path
        )

        mapped_d_label_reaug = map_labels_to_diseases(d_label_reaug, ctx.label_to_related_disease).reshape(-1, 1)
        mapped_q_label_reaug = map_labels_to_diseases(q_label_reaug, ctx.label_to_related_disease).reshape(-1, 1)

        mapped_d_reaug_idx = map_indices(int(P / 2), d_reaug_idx).reshape(-1, 2)
        mapped_q_reaug_idx = map_indices(int(P / 2), q_reaug_idx).reshape(-1, 2)

        d_report_diseases = ctx.report_disease[mapped_d_reaug_idx[:, 0], :]
        q_report_diseases = ctx.report_disease[mapped_q_reaug_idx[:, 0], :]
        d_diseases_summary = get_diseases_summary(d_report_diseases, mapped_d_label_reaug)
        q_diseases_summary = get_diseases_summary(q_report_diseases, mapped_q_label_reaug)
        d_diseases_average = get_diseases_average(d_diseases_summary)
        q_diseases_average = get_diseases_average(q_diseases_summary)

        threshold = 0.0
        d_diseases_average = np.where(d_diseases_average <= threshold, 0, 1).astype(np.int64)
        q_diseases_average = np.where(q_diseases_average <= threshold, 0, 1).astype(np.int64)

        d_label_reaug = d_label_reaug.reshape(-1)
        q_label_reaug = q_label_reaug.reshape(-1)

        d_names = np.array([all_diseases[i] for i, v in enumerate(d_diseases_average) if v == 1])
        q_names = np.array([all_diseases[i] for i, v in enumerate(q_diseases_average) if v == 1])
        missing = np.setdiff1d(d_names, q_names)
        adding = np.setdiff1d(q_names, d_names)

        reaug_state = get_caption(adding, missing) if (adding.size > 0 or missing.size > 0) else "nothing has changed."
        tokens = nltk.word_tokenize(reaug_state.lower())
        token_ids = [ctx.word_to_idx.get(w, 0) for w in tokens]
        reaug_state_list = get_label(token_ids, 90).astype(np.int64)

        rag_features_list.append([d_reaug_features, q_reaug_features])
        rag_label_list.append([d_label_reaug, q_label_reaug])
        diseases_avg_list.append([d_diseases_average, q_diseases_average])
        reaug_state_list_list.append(reaug_state_list)
        rag_bb_list.append([d_reaug_bb, q_reaug_bb])
        rag_semantic_adj_list.append([rag_d_semantic_adj, rag_q_semantic_adj])
        rag_adj_matrix_list.append([rag_d_adj_matrix, rag_q_adj_matrix])

    return (
        rag_features_list,
        rag_label_list,
        diseases_avg_list,
        reaug_state_list_list,
        rag_bb_list,
        rag_semantic_adj_list,
        rag_adj_matrix_list,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split-size", type=int, default=5000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--num-sources-per-image", type=int, default=52)
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_root

    paths = {
        "report_disease": os.path.join(data_dir, "train_report_disease.pkl"),
        "splits": os.path.join(data_dir, "splits_mimic_VQA.json"),
        "faiss_nns": os.path.join(data_dir, "cmb_vqadiff_nns.hdf5"),
        "vqa_h5": os.path.join(data_dir, "VQA_mimic_dataset.h5"),
        "cmb_feats": os.path.join(data_dir, "train_cmb_bbox_di_feats.hdf5"),
        "vocab": os.path.join(data_dir, "vocab_mimic_VQA.json"),
    }

    with open(paths["report_disease"], "rb") as f:
        report_disease = pickle.load(f)

    splits = json.load(open(paths["splits"], "r"))
    for split in ["train", "val", "test"]:

        split_idxs = splits[split]

        with h5py.File(paths["faiss_nns"], "r") as f:
            B = f["faiss_nns"].shape[0]
            faiss_nns = f["faiss_nns"][:].reshape(int(B / 26), 26, -1)

        with h5py.File(paths["vqa_h5"], "r") as f:
            feature_idx = f["feature_idx"][:]

        with h5py.File(paths["cmb_feats"], "r") as f:
            faiss_d = f["image_features"].shape[-1]
            faiss_features = f["image_features"][:, :26, :].reshape(-1, faiss_d)
            faiss_label = f["bbox_label"][:, :26, :].reshape(-1)
            faiss_bb = f["image_bb"][:, :26, :].reshape(-1, 4)

            ana_feat = f["image_features"][:, 26:, :].reshape(-1, faiss_d)
            ana_label = f["bbox_label"][:, 26:, :].reshape(-1)
            ana_bb = f["image_bb"][:, 26:, :].reshape(-1, 4)

        word_to_idx = json.load(open(paths["vocab"], "r"))

        # Precompute mapping
        ans_loc_dict = get_chest_xray_findings()
        disease_dict = get_disease_loc_dict()
        disease_list = [k for k in disease_dict.keys()]
        label_to_related_disease = map_findings_to_indices(ans_loc_dict, disease_list)

        ctx = Context(
            faiss_nns=faiss_nns,
            feature_idx=feature_idx,
            faiss_features=faiss_features,
            faiss_label=faiss_label,
            faiss_bb=faiss_bb,
            ana_feat=ana_feat,
            ana_label=ana_label,
            ana_bb=ana_bb,
            report_disease=report_disease,
            word_to_idx=word_to_idx,
            label_to_related_disease=label_to_related_disease,
            k=args.k,
        )

        print("num split indices:", len(split_idxs))
        print("k:", args.k)

        split_size = args.split_size
        iterations = len(split_idxs) // split_size + 1
        times = 0

        for it in range(iterations):
            print("it", it)
            sub = split_idxs[it * split_size : (it + 1) * split_size]
            if len(sub) == 0:
                continue

            results = [process_chunk(ctx, sub, args.feat_dir)]

            rag_features_list = []
            rag_label_list = []
            diseases_avg_list = []
            reaug_state_list_list = []
            rag_bb_list = []
            rag_semantic_adj_list = []
            rag_adj_matrix_list = []

            for r in results:
                (rf, rl, da, rs, rbb, rsa, ram) = r
                rag_features_list.extend(rf)
                rag_label_list.extend(rl)
                diseases_avg_list.extend(da)
                reaug_state_list_list.extend(rs)
                rag_bb_list.extend(rbb)
                rag_semantic_adj_list.extend(rsa)
                rag_adj_matrix_list.extend(ram)

            rag_features_arr = np.asarray(rag_features_list, dtype=np.float32)
            rag_label_arr = np.asarray(rag_label_list, dtype=np.int64)
            diseases_avg_arr = np.asarray(diseases_avg_list, dtype=np.int64)
            reaug_state_arr = np.asarray(reaug_state_list_list, dtype=np.int64)
            rag_bb_arr = np.asarray(rag_bb_list, dtype=np.float64)
            rag_semantic_adj_arr = np.asarray(rag_semantic_adj_list, dtype=np.int64)
            rag_adj_matrix_arr = np.asarray(rag_adj_matrix_list, dtype=np.int64)

            print("rag_features_arr", rag_features_arr.shape)
            length = len(rag_features_arr)

            save_h5(
                filename=os.path.join(data_dir, f"retrieval_cmb_{split}.hdf5"),
                reaug_features=rag_features_arr,
                rag_label=rag_label_arr,
                diseases_avg=diseases_avg_arr,
                reaug_state_list=reaug_state_arr,
                rag_bb=rag_bb_arr,
                rag_semantic_adj=rag_semantic_adj_arr,
                rag_adj_matrix=rag_adj_matrix_arr,
                test_num_sources_per_image=args.num_sources_per_image,
                times=times,
                length=length,
                k=args.k,
            )

            times += 1
            del results
            gc.collect()

    print("finished")


if __name__ == "__main__":
    main()

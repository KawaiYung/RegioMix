import os
import json
import numpy as np
import random
import time
import pickle
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
import json
import nltk


nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


def get_chest_xray_findings():
    chest_xray_findings = {
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
            "Hematoma",
            "Fracture",
        ],
        "right upper lung zone": ["Pneumothorax"],
        "right mid lung zone": ["Pneumothorax", "Pleural Effusion"],
        "right lower lung zone": ["Pleural Effusion"],
        "right hilar structures": ["Hilar Congestion"],
        "right apical zone": ["Pneumothorax"],
        "right costophrenic angle": [
            "Blunting of the Costophrenic Angle",
            "Pleural Effusion",
        ],
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
        "left costophrenic angle": [
            "Blunting of the Costophrenic Angle",
            "Pleural Effusion",
        ],
        "left hemidiaphragm": ["Gastric Distention", "Pleural Thickening", "Hernia"],
        "trachea": ["Infection", "scoliosis"],
        "right clavicle": ["Hematoma", "Fracture"],
        "left clavicle": ["Hematoma", "Fracture"],
        "aortic arch": [
            "Calcification",
            "Tortuosity of the Descending Aorta",
            "Tortuosity of the Thoracic Aorta",
        ],
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
    return chest_xray_findings


def get_disease_loc_dict():
    ans_loc_dict = {}
    ans_loc_dict["hilar congestion"] = [
        "right hilar structures",
        "left hilar structures",
    ]
    ans_loc_dict["vascular congestion"] = ["right lung", "left lung"]
    ans_loc_dict["blunting of the costophrenic angle"] = [
        "right costophrenic angle",
        "left costophrenic angle",
    ]
    ans_loc_dict["calcification"] = ["aortic arch", "upper mediastinum"]
    ans_loc_dict["pneumonia"] = ["right lung", "left lung"]
    ans_loc_dict["granuloma"] = ["right lung", "left lung"]
    ans_loc_dict["gastric distention"] = ["left lower lung zone"]  # Gastrointestinal
    ans_loc_dict["heart failure"] = ["cardiac silhouette", "right lung", "left lung"]
    ans_loc_dict["atelectasis"] = ["right lower lung zone", "left lower lung zone"]
    ans_loc_dict["scoliosis"] = ["spine"]
    ans_loc_dict["pneumothorax"] = ["right upper lung zone", "left upper lung zone"]
    ans_loc_dict["pleural thickening"] = [
        "right lower lung zone",
        "left lower lung zone",
    ]
    ans_loc_dict["infection"] = ["right lung", "left lung"]
    ans_loc_dict["enlargement of the cardiac silhouette"] = ["cardiac silhouette"]
    ans_loc_dict["hypertensive heart disease"] = ["cardiac silhouette"]
    ans_loc_dict["hernia"] = ["diaphragm regions"]  # Specific category depends on hernia type
    ans_loc_dict["tortuosity of the descending aorta"] = ["aortic arch"]
    ans_loc_dict["hypoxemia"] = ["right lung", "left lung"]  # Respiratory or systemic
    ans_loc_dict["thymoma"] = ["upper mediastinum"]
    ans_loc_dict["consolidation"] = ["right lung", "left lung"]
    ans_loc_dict["hematoma"] = ["other"]  # Depends on location
    ans_loc_dict["fracture"] = [
        "right clavicle",
        "left clavicle",
        "right lung",
        "left lung",
    ]
    ans_loc_dict["tortuosity of the thoracic aorta"] = ["aortic arch"]
    ans_loc_dict["pneumomediastinum"] = ["mediastinum"]
    ans_loc_dict["pleural effusion"] = [
        "right costophrenic angle",
        "left costophrenic angle",
    ]
    ans_loc_dict["contusion"] = ["other"]  # Depends on location
    ans_loc_dict["lung opacity"] = ["right lung", "left lung"]
    ans_loc_dict["emphysema"] = ["right upper lung zone", "left upper lung zone"]
    ans_loc_dict["air collection"] = ["right lung", "left lung"]
    ans_loc_dict["cardiomegaly"] = ["cardiac silhouette"]
    ans_loc_dict["edema"] = [
        "right lung",
        "left lung",
        "edema",
    ]  # If related to pulmonary edema
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


def get_group_indices(ans_loc_dict, detection_dict):
    # Convert the values of ans_loc_dict to a list for easy indexing
    ans_loc_values = list(ans_loc_dict.values())

    group_indices = []
    for detection_key in detection_dict:
        # Get the group of the current detection_dict key
        group = detection_dict[detection_key]

        # Find all indices in ans_loc_dict that have the same group
        indices = [i for i, v in enumerate(ans_loc_values) if v == group]

        # Append the indices to the group_indices list
        group_indices.append(indices)

    return group_indices


def map_findings_to_indices(chest_xray_findings, findings_list):
    # Convert all elements in findings_list to lowercase for case-insensitive comparison
    findings_list_lower = [f.lower() for f in findings_list]

    # Initialize the result array
    result = []

    # Iterate over the dictionary
    for key in chest_xray_findings:
        # Initialize the array for the current key
        current_key_indices = []

        # Iterate over the findings for the current key
        for finding in chest_xray_findings[key]:
            # Convert finding to lowercase
            finding_lower = finding.lower()

            # Find the index in findings_list, if it exists
            if finding_lower in findings_list_lower:
                index = findings_list_lower.index(finding_lower)
                current_key_indices.append(index)

        # Append the indices array for the current key to the result array
        result.append(current_key_indices)

    return result


# import faiss.contrib.torch_utils

# def map_indices(original_shape, retrieved_indices):
#     # Flatten the retrieved indices if it's a 2D array
#     if retrieved_indices.ndim > 1:
#         retrieved_indices = retrieved_indices.flatten()

#     # Calculate the corresponding indices in the original tensor shape
#     dim1_size = original_shape[1]
#     mapped_indices = np.stack((retrieved_indices // dim1_size, retrieved_indices % dim1_size), axis=1)

#     return mapped_indices


ans_loc_dict = get_chest_xray_findings()
disease_dict = get_disease_loc_dict()
disease_list = [f"{key}" for key, value in disease_dict.items()]
label_to_related_disease = map_findings_to_indices(ans_loc_dict, disease_list)
# print(label_to_related_disease)
# label_to_related_disease = [[i for i in range(31)] for _ in range(26)]
# print(label_to_related_disease)
# label_to_related_disease = [[18, 0, 22, 23],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [29, 13, 14, 7, 1, 23, 30],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [9, 17, 20, 21, 25],
#     [24, 2, 6, 10, 26, 19, 11, 15, 8, 4, 5, 3, 27, 12],
#     [29, 13, 14, 7, 1, 23, 30, 8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [29, 13, 14, 7, 1, 23, 30],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [29, 13, 14, 7, 1, 23, 30],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [24, 2, 6, 10, 26, 19, 11, 15],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12],
#     [24, 2, 6, 10, 26, 19, 11, 15, 8, 4, 5, 3, 27, 12],
#     [24, 2, 6, 10, 26, 19, 11, 15],
#     [9, 17, 20, 21, 25],
#     [8, 4, 26, 5, 3, 19, 11, 27, 12]]


def get_caption(
    adding,
    dropping,
):
    if len(adding) == 1:
        output1 = "the main image has an additional finding of"
    elif len(adding) > 1:
        output1 = "the main image has additional findings of"
    elif len(adding) == 0:
        output1 = ""
    for item in adding:
        if item == adding[-1] and len(adding) != 1:
            output1 = output1 + " and " + item
        else:
            if len(adding) == 1:
                output1 = output1 + " " + item
            else:
                output1 = output1 + " " + item + ","
    if len(adding) != 0:
        output1 = output1 + " than the reference image. "

    if len(dropping) == 1:
        output2 = "the main image is missing the finding of"
    elif len(dropping) > 1:
        output2 = "the main image is missing the findings of"
    elif len(dropping) == 0:
        output2 = ""
    for item in dropping:
        if item == dropping[-1] and len(dropping) != 1:
            output2 = output2 + " and " + item
        else:
            if len(dropping) == 1:
                output2 = output2 + " " + item
            else:
                output2 = output2 + " " + item + ","

    if len(dropping) != 0:
        output2 = output2 + " than the reference image. "
    return output1 + output2


def get_label(caption_list, max_seq):
    output = np.zeros(max_seq)
    output[: len(caption_list)] = np.array(caption_list)
    return output


def map_labels_to_diseases(label):
    # # Convert label_to_related_disease to an array of objects
    # disease_array = np.array(label_to_related_disease, dtype=object)

    # Create an empty object array to hold lists
    disease_array = np.empty(len(label_to_related_disease), dtype=object)

    # Populate the array with the data lists
    for i in range(len(label_to_related_disease)):
        disease_array[i] = label_to_related_disease[i]

    # Use advanced indexing to map the label array
    mapped_array = disease_array[label.squeeze()]
    # Reshape to match the original label array
    return mapped_array.reshape(label.shape[0], label.shape[1], -1)


def get_diseases_summary(result_array, int_mapped):
    """
    Process the result_array based on indices provided in int_mapped.
    Each element in result_array not indexed by int_mapped is set to -1.

    :param result_array: numpy array of shape (n, m)
    :param int_mapped: numpy array of indices, shape (n, 1) where each entry is a list of indices
    :return: numpy array of the same shape as result_array with processed values
    """
    processed_array = np.full(result_array.shape, -1)
    for row_idx, (indices_row, result_row) in enumerate(zip(int_mapped, result_array)):
        indices = np.array(indices_row[0], dtype=int)  # Convert indices to integer type
        processed_array[row_idx, indices] = [result_row[i] for i in indices]
    return processed_array


def get_diseases_average(array):
    """
    Compute the average of each column in the array, excluding negative values.
    If a column contains only negative values, the average for that column is set to 0.

    :param array: numpy array
    :return: numpy array containing the average for each column
    """
    non_negative_mask = np.where(array >= 0, 1, 0)
    sum_values = np.sum(np.where(array >= 0, array, 0), axis=0)
    count_non_negatives = np.sum(non_negative_mask, axis=0)

    count_non_negatives[count_non_negatives == 0] = 1
    average = sum_values / count_non_negatives
    average[count_non_negatives == 1] = 0

    return average


def map_indices(shape, indices):
    # Using PyTorch operations for vectorization across all indices
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    # Using NumPy operations for vectorization
    flat_indices = indices.reshape(-1)
    return np.stack((flat_indices // shape, flat_indices % shape), axis=-1).reshape(*indices.shape, 2)


class FaissIndex:
    _instance = None

    @classmethod
    def get_instance(cls, cfg):
        if cls._instance is None:
            cls._instance = cls._create_index(cfg)
        return cls._instance

    @staticmethod
    def _create_index(cfg):

        train_cmb_bbox_di_feats_path = cfg.data.train_cmb_bbox_di_feats_path
        train_cmb_hf3 = h5py.File(train_cmb_bbox_di_feats_path, "r")
        # (img_num (377110), bb_num (26), d(1024)) -> (img_num*bb_num, d)
        faiss_d = train_cmb_hf3["image_features"].shape[-1]
        bb_features = train_cmb_hf3["image_features"][:, :26, :].reshape(-1, faiss_d)
        ana_feat = train_cmb_hf3["image_features"][:, 26:, :].reshape(-1, faiss_d)
        faiss_nns_path = cfg.data.faiss_nns_path
        faiss_nns_h5 = h5py.File(faiss_nns_path, "r")
        B = faiss_nns_h5["faiss_nns"].shape[0]
        faiss_nns = faiss_nns_h5["faiss_nns"][:].reshape(int(B / 26), 26, -1)

        return (bb_features, ana_feat, faiss_nns)


class RCCDataset_mimic(Dataset):
    shapes = set(["ball", "block", "cube", "cylinder", "sphere"])
    sphere = set(["ball", "sphere"])
    cube = set(["block", "cube"])
    cylinder = set(["cylinder"])

    colors = set(["red", "cyan", "brown", "blue", "purple", "green", "gray", "yellow"])

    materials = set(["metallic", "matte", "rubber", "shiny", "metal"])
    rubber = set(["matte", "rubber"])
    metal = set(["metal", "metallic", "shiny"])

    type_to_label = {"change": 0, "no_change": 1}

    def __init__(self, cfg, split):
        self.cfg = cfg
        print("Speaker Dataset loading vocab json file: ", cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        # if cfg.data.use_reaug:
        #     print('Speaker Dataset loading faiss vocab json file: ', cfg.faiss.vocab_json)
        #     self.vocab_json = cfg.faiss.vocab_json

        self.word_to_idx = json.load(open(self.vocab_json, "r"))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.idx_to_word) + 1
        print("vocab size is ", self.vocab_size)

        self.splits = json.load(open(cfg.data.splits_json, "r"))
        self.split = split

        if split == "train":
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
            self.split_idxs = self.splits["train"]
            self.num_samples = len(self.split_idxs)
            if cfg.data.train.max_samples is not None:
                self.num_samples = min(cfg.data.train.max_samples, self.num_samples)
        elif split == "val":
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
            self.split_idxs = self.splits["val"]
            self.num_samples = len(self.split_idxs)
            if cfg.data.val.max_samples is not None:
                self.num_samples = min(cfg.data.val.max_samples, self.num_samples)
        elif split == "test":
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits["test"]
            self.num_samples = len(self.split_idxs)
            if cfg.data.test.max_samples is not None:
                self.num_samples = min(max_samples, self.num_samples)
        elif split == "all":
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = self.splits["train"] + self.splits["val"] + self.splits["test"]
            self.num_samples = len(self.split_idxs)
        else:
            raise Exception("Unknown data split %s" % split)

        print("Dataset size for %s: %d" % (split, self.num_samples))

        # load in the sequence data
        self.h5_label_file = h5py.File(cfg.data.h5_label_file, "r")
        self.labels = self.h5_label_file["answers"][:]  # just gonna load...
        self.questions = self.h5_label_file["questions"][:]
        # self.neg_labels = self.h5_label_file['neg_labels'][:]
        self.pos = self.h5_label_file["pos"][:]
        # self.pos = self.h5_label_file['answers'][:][:,:20]
        seq_size = self.labels.shape
        # self.neg_pos = self.h5_label_file['neg_pos'][:]
        self.max_seq_length = seq_size[1]
        # self.label_start_idx = self.h5_label_file['label_start_idx'][:]
        # self.label_end_idx = self.h5_label_file['label_end_idx'][:]
        self.label_start_idx = np.arange(len(self.pos)).reshape(len(self.pos), 1)
        self.label_end_idx = self.label_start_idx + 1
        self.feature_idx = self.h5_label_file["feature_idx"][:]

        # self.neg_label_start_idx = self.h5_label_file['neg_label_start_idx'][:]
        # self.neg_label_end_idx = self.h5_label_file['neg_label_end_idx'][:]
        print("Max sequence length is %d" % self.max_seq_length)
        self.h5_label_file.close()

        path4 = cfg.data.cmb_bbox_di_feats_path
        # self.hf1 = h5py.File(path1, 'r')
        # self.hf2 = h5py.File(path2, 'r')
        # self.hf3 = h5py.File(path3, 'r')
        self.hf4 = h5py.File(path4, "r")
        # self.features = self.hf1['image_features']
        # self.features2 = self.hf2['image_features']
        # self.features3 = self.hf3['image_features']
        self.features4 = self.hf4["image_features"]
        self.bb = self.hf4["image_bb"]
        self.node_one_num = int(len(self.features4[0]) / 2)
        assert self.node_one_num == 26  # or go to modify nongt_dim
        self.adj = self.hf4["image_adj_matrix"]
        self.sem_adj = self.hf4["semantic_adj_matrix"]

        if cfg.data.feature_mode == "mode0":
            qestions_path = os.path.join("data", "../data/mimic_pair_questions.csv")
            self.pd = pd.read_csv(qestions_path)
            path = "/home/xinyue/dataset/mimic/mimic_all.csv"
            with open(path, "r") as f:
                self.mimic_all = pd.read_csv(f)
        # self.hf1.close()
        # self.hf2.close()

        if cfg.data.use_reaug:
            # faiss
            self.reaug_k = cfg.faiss.reaug_k
            if split == "train":
                self.reaug_k = self.reaug_k + 1  # the first retrived element in train is itself

            if split == "train":
                retrieval = h5py.File(
                    cfg.data.retrieval_cmb_train_path,
                    "r",
                )
            elif split == "test":
                retrieval = h5py.File(
                    cfg.data.retrieval_cmb_test_path,
                    "r",
                )
            (self.rag_bb_feat, self.rag_ana_feat, self.faiss_nns) = FaissIndex.get_instance(cfg)

            self.rag_label = retrieval["rag_label"]
            self.diseases_avg = retrieval["diseases_avg"]
            self.reaug_state_list = retrieval["reaug_state_list"]
            self.rag_bb = retrieval["rag_bb"]
            self.rag_semantic_adj = retrieval["rag_semantic_adj"]
            self.rag_adj_matrix = retrieval["rag_adj_matrix"]

    def fill_adj(self, adj, len, multiplier):
        for i in range(multiplier):
            for j in range(multiplier):
                if i == 0 and j == 0:
                    continue
                adj[i * len : (i + 1) * len, j * len : (j + 1) * len] = adj[:len, :len]
        # adj[len:2*len, :len] = adj[:len, :len]
        # adj[:len, len:2*len] = adj[:len, :len]
        # adj[len:2*len, len:2*len] = adj[:len, :len]
        return adj

    def move_adj(self, adj, len, mode="3to2"):
        # move the 3rd adj to 2nd position
        if mode == "3to2":
            adj[len : 2 * len] = adj[2 * len : 3 * len]
            adj[:, len : 2 * len] = adj[:, 2 * len : 3 * len]
        elif mode == "3to1":
            adj[:len] = adj[2 * len : 3 * len]
            adj[:, :len] = adj[:, 2 * len : 3 * len]
        return adj

    def get_image(self, index):
        study_id = self.pd.iloc[index]["study_id"]
        dicom_id = self.mimic_all[self.mimic_all["study_id"] == int(study_id)]["dicom_id"].values[0]
        file_path = "/home/xinyue/dataset/mimic-cxr-png/%s.png" % str(dicom_id)
        image = Image.open(file_path)
        image = image.resize((128, 128))
        image = np.array(image)
        return image

    def decode_sequence(self, seq):
        ix_to_word = self.idx_to_word
        N, D = seq.size()
        out = []
        for i in range(N):
            txt = ""
            for j in range(D):
                ix = seq[i, j]
                if ix > 0:
                    if j >= 1:
                        txt = txt + " "
                    txt = txt + ix_to_word[ix.item()]
                else:
                    break
            out.append(txt)
        return out

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        random.seed(1111)

        img_idx = self.split_idxs[index]  # pair index
        # feature order: ana, location
        if self.cfg.data.feature_mode == "both" or self.cfg.data.feature_mode == "location":
            d_feature = self.features4[self.feature_idx[img_idx, 0]]
            q_feature = self.features4[self.feature_idx[img_idx, 1]]
            d_bbox = torch.tensor([1])
            q_bbox = torch.tensor([1])

            d_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 0]]).double()
            q_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 1]]).double()
            d_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 0]]).double()
            q_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 1]]).double()
            d_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 0]]).double()
            q_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 1]]).double()
        elif self.cfg.data.feature_mode == "single_ana":
            d_feature = self.features4[self.feature_idx[img_idx, 0]][: self.node_one_num]
            q_feature = self.features4[self.feature_idx[img_idx, 1]][: self.node_one_num]
            d_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 0]]).double()
            q_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 1]]).double()
            d_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 0]]).double()
            q_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 1]]).double()
            d_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 0]]).double()[: self.node_one_num]
            q_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 1]]).double()[: self.node_one_num]
        elif self.cfg.data.feature_mode == "single_loc":
            d_feature = self.features4[self.feature_idx[img_idx, 0]][-self.node_one_num :]
            q_feature = self.features4[self.feature_idx[img_idx, 1]][-self.node_one_num :]
            d_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 0]]).double()
            q_adj_matrix = torch.from_numpy(self.adj[self.feature_idx[img_idx, 1]]).double()
            d_adj_matrix = self.move_adj(d_adj_matrix, self.node_one_num, mode="3to1")
            q_adj_matrix = self.move_adj(q_adj_matrix, self.node_one_num, mode="3to1")
            d_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 0]]).double()
            q_sem_adj_matrix = torch.from_numpy(self.sem_adj[self.feature_idx[img_idx, 1]]).double()
            d_sem_adj_matrix = self.move_adj(d_sem_adj_matrix, self.node_one_num, mode="3to1")
            q_sem_adj_matrix = self.move_adj(q_sem_adj_matrix, self.node_one_num, mode="3to1")
            d_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 0]]).double()[: self.node_one_num]
            q_bb = torch.from_numpy(self.bb[self.feature_idx[img_idx, 1]]).double()[: self.node_one_num]
        #
        elif self.cfg.data.feature_mode == "mode0":
            image1 = self.get_image(self.feature_idx[img_idx, 0])
            image2 = self.get_image(self.feature_idx[img_idx, 1])
            d_feature = image1
            q_feature = image2
            d_adj_matrix = 0
            q_adj_matrix = 0
            d_sem_adj_matrix = 0
            q_sem_adj_matrix = 0
            d_bb = 0
            q_bb = 0

        P, D = d_feature.shape
        d_reaug_features = q_reaug_features = torch.zeros([int(P / 2), self.reaug_k, d_feature.shape[-1]])
        if self.cfg.data.use_reaug == True:
            # Only query with disease features
            if self.cfg.hpc.is_hpc:
                reaug_features = self.rag_features[index]
                d_reaug_features = reaug_features[0, :, : self.reaug_k, :]
                q_reaug_features = reaug_features[1, :, : self.reaug_k, :]

            else:
                d_reaug_idx = self.faiss_nns[self.feature_idx[index, 0], :, : self.reaug_k]

                q_reaug_idx = self.faiss_nns[self.feature_idx[index, 1], :, : self.reaug_k]

                d_reaug_bbox = self.rag_bb_feat[d_reaug_idx, :]
                q_reaug_bbox = self.rag_bb_feat[q_reaug_idx, :]
                d_reaug_ana = self.rag_ana_feat[d_reaug_idx, :]
                q_reaug_ana = self.rag_ana_feat[q_reaug_idx, :]
                d_reaug_features = np.concatenate((d_reaug_ana, d_reaug_bbox), axis=0)
                q_reaug_features = np.concatenate((q_reaug_ana, q_reaug_bbox), axis=0)

            reaug_bb = self.rag_bb[index]
            d_reaug_bb = reaug_bb[0, : self.reaug_k, :, :]
            q_reaug_bb = reaug_bb[1, : self.reaug_k, :, :]

            diseases_average = self.diseases_avg[index]
            d_diseases_average = diseases_average[0, :]
            q_diseases_average = diseases_average[1, :]

            bb_label_reaug = self.rag_label[index]
            d_bb_label_reaug = bb_label_reaug[0]
            q_bb_label_reaug = bb_label_reaug[1]

            reaug_state_list = self.reaug_state_list[index]
            reaug_state_list = np.insert(reaug_state_list, 0, 1)

            rag_semantic_adj = self.rag_semantic_adj[index]
            d_rag_semantic_adj = torch.from_numpy(rag_semantic_adj[0, : self.reaug_k, :, :]).double()
            q_rag_semantic_adj = torch.from_numpy(rag_semantic_adj[1, : self.reaug_k, :, :]).double()

            rag_adj_matrix = self.rag_adj_matrix[index]
            d_rag_adj_matrix = torch.from_numpy(rag_adj_matrix[0, : self.reaug_k, :, :]).double()
            q_rag_adj_matrix = torch.from_numpy(rag_adj_matrix[1, : self.reaug_k, :, :]).double()

            if self.split == "train":
                d_reaug_features = d_reaug_features[:, 1:, :]  # Ignore first retrived k
                q_reaug_features = q_reaug_features[:, 1:, :]
                d_reaug_bb = d_reaug_bb[1:, :, :]
                q_reaug_bb = q_reaug_bb[1:, :, :]
                d_rag_semantic_adj = d_rag_semantic_adj[1:, :, :]
                q_rag_semantic_adj = q_rag_semantic_adj[1:, :, :]
                d_rag_adj_matrix = d_rag_adj_matrix[1:, :, :]
                q_rag_adj_matrix = q_rag_adj_matrix[1:, :, :]

        # Fetch sequence labels
        ix1 = self.label_start_idx[img_idx]
        ix2 = self.label_end_idx[img_idx]
        n_cap = ix2 - ix1 + 1

        seq = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        pos = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        if n_cap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, : self.max_seq_length] = self.labels[ixl, : self.max_seq_length]
                pos[q, : self.max_seq_length] = self.pos[ixl, : self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            seq[:, : self.max_seq_length] = self.labels[ixl : ixl + self.seq_per_img, : self.max_seq_length]
            pos[:, : self.max_seq_length] = self.pos[ixl : ixl + self.seq_per_img, : self.max_seq_length]

        # Generate masks
        mask = np.zeros_like(seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, seq)))
        for ix, row in enumerate(mask):
            row[: nonzeros[ix]] = 1

        # neg_mask = np.zeros_like(neg_seq)
        # nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, neg_seq)))
        # for ix, row in enumerate(neg_mask):
        #     row[:nonzeros[ix]] = 1
        question = self.questions[ixl]
        question_mask = 0
        target_question = [
            2,
            25,
            26,
            27,
            28,
            7,
            29,
            30,
            10,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        # Check if all elements in question are equal to the target_question
        if (question == target_question).all():
            question_mask = 1
        # (number of detection boxes, topk retrivel, feature dimention)
        return (
            d_feature,
            q_feature,
            seq,
            pos,
            mask,
            img_idx,
            d_adj_matrix,
            q_adj_matrix,
            d_sem_adj_matrix,
            q_sem_adj_matrix,
            d_bb,
            q_bb,
            question,
            d_reaug_features,
            q_reaug_features,
            d_bb_label_reaug,
            q_bb_label_reaug,
            d_diseases_average,
            q_diseases_average,
            reaug_state_list,
            d_reaug_bb,
            q_reaug_bb,
            d_rag_semantic_adj,
            q_rag_semantic_adj,
            d_rag_adj_matrix,
            q_rag_adj_matrix,
            question_mask,
        )

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


def rcc_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    q_feat_batch = transposed[1]
    seq_batch = default_collate(transposed[2])
    pos_batch = default_collate(transposed[3])
    mask_batch = default_collate(transposed[4])
    index_batch = default_collate(transposed[5])
    d_adj_matrix = default_collate(transposed[6])
    q_adj_matrix = default_collate(transposed[7])
    d_sem_adj_matrix = default_collate(transposed[8])
    q_sem_adj_matrix = default_collate(transposed[9])
    d_bb = default_collate(transposed[10])
    q_bb = default_collate(transposed[11])
    question = default_collate(transposed[12])
    d_reaug_features_batch = default_collate(transposed[13])
    q_reaug_features_batch = default_collate(transposed[14])
    d_bb_label_reaug = default_collate(transposed[15])
    q_bb_label_reaug = default_collate(transposed[16])
    d_diseases_average = default_collate(transposed[17])
    q_diseases_average = default_collate(transposed[18])
    reaug_state_list = default_collate(transposed[19])
    d_reaug_bb = default_collate(transposed[20])
    q_reaug_bb = default_collate(transposed[21])
    rag_d_semantic_adj = default_collate(transposed[22])
    rag_q_semantic_adj = default_collate(transposed[23])
    rag_d_adj_matrix = default_collate(transposed[24])
    rag_q_adj_matrix = default_collate(transposed[25])
    question_mask = default_collate(transposed[26])
    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    # d_img_batch = transposed[11]
    # n_img_batch = transposed[12]
    # q_img_batch = transposed[13]
    return (
        d_feat_batch,
        q_feat_batch,
        seq_batch,
        pos_batch,
        mask_batch,
        index_batch,
        d_adj_matrix,
        q_adj_matrix,
        d_sem_adj_matrix,
        q_sem_adj_matrix,
        d_bb,
        q_bb,
        question,
        d_reaug_features_batch,
        q_reaug_features_batch,
        d_bb_label_reaug,
        q_bb_label_reaug,
        d_diseases_average,
        q_diseases_average,
        reaug_state_list,
        d_reaug_bb,
        q_reaug_bb,
        rag_d_semantic_adj,
        rag_q_semantic_adj,
        rag_d_adj_matrix,
        rag_q_adj_matrix,
        question_mask,
    )


class RCCDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs["collate_fn"] = rcc_collate
        super().__init__(dataset, **kwargs)


if __name__ == "__main__":
    from configs.config import cfg, merge_cfg_from_file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic/dynamic_change_pos_mimic.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--visualize_every", type=int, default=10)
    parser.add_argument("--setting", type=str, default="mode2")
    parser.add_argument(
        "--graph",
        type=str,
        default="all",
        choices=["implicit", "semantic", "spatial", "all"],
    )
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_fold", type=str, default="mode2_location_0.0001_2022-05-19-00-18-37")
    parser.add_argument("--snapshot", type=int, default=5000)
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="location",
        choices=["both", "coords", "location", "single_ana", "single_loc"],
    )  # both means ana+coords+location.
    parser.add_argument("--seed", type=int, default=1113)

    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)

    dataset = RCCDataset_mimic(cfg, "test")
    print(dataset[10])

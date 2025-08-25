import numpy as np
import math
import pickle
import os
import json
import pandas as pd
import h5py


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (
        (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)
        + (gt_box[2] - gt_box[0] + 1.0) * (gt_box[3] - gt_box[1] + 1.0)
        - inters
    )

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_center(bb1):
    c1 = [(bb1[2] + bb1[0]) / 2, (bb1[3] + bb1[1]) / 2]
    return c1


def get_distance(bb1, bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    dx = np.abs(c2[0] - c1[0])
    dy = np.abs(c2[1] - c1[1])
    d = np.sqrt(np.square(dx) + np.square(dy))
    return d


def get_angle(coor):
    x1, y1, x2, y2 = coor
    angle = math.atan2(y2 - y1, x2 - x1) / math.pi * 180
    if angle < 0:
        angle += 360
    return angle


def cal_angle(bb1, bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    return get_angle(c1 + c2)


def bbox_relation_type(bb1, bb2, lx=1024, ly=1024):
    if bb1[0] < bb2[0] and bb1[1] < bb2[1] and bb1[2] > bb2[2] and bb1[3] > bb2[3]:
        return 1
    elif bb1[0] > bb2[0] and bb1[1] > bb2[1] and bb1[2] < bb2[2] and bb1[3] < bb2[3]:
        return 2
    elif get_iou(bb1, bb2) >= 0.5:
        return 3
    elif get_distance(bb1, bb2) >= (lx + ly) / 3:
        return 0
    angle = cal_angle(bb1, bb2)
    return math.ceil(angle / 45) + 3


def reverse_type(type):
    if type == 0:
        return 0
    elif type == 1:
        return 2
    elif type == 2:
        return 1
    elif type == 3:
        return 3
    elif type == 4:
        return 8
    elif type == 5:
        return 9
    elif type == 6:
        return 10
    elif type == 7:
        return 11
    elif type == 8:
        return 4
    elif type == 9:
        return 5
    elif type == 10:
        return 6
    elif type == 11:
        return 7


def get_adj_matrix(bboxes, adj_matrix=None):

    num_pics = len(bboxes)
    n = len(bboxes[0])
    if adj_matrix is None:
        adj_matrix = np.zeros([num_pics, 100, 100], int)
    for idx in range(num_pics):
        bbs = bboxes[idx]
        for i in range(n):
            for j in range(i, n):
                if adj_matrix[idx, i, j] != 0:
                    continue
                type = bbox_relation_type(bbs[i], bbs[j])
                adj_matrix[idx, i, j] = type
                adj_matrix[idx, j, i] = reverse_type((type))
    return adj_matrix


def get_kg_ana_only():
    # more simpler names
    kg_dict = {}
    # anatomical part
    kg_dict["right lung"] = "Lung"
    kg_dict["right upper lung zone"] = "Lung"
    kg_dict["right mid lung zone"] = "Lung"
    kg_dict["right lower lung zone"] = "Lung"
    kg_dict["right hilar structures"] = "Lung"
    kg_dict["right apical zone"] = "Lung"
    kg_dict["right costophrenic angle"] = "Pleural"
    kg_dict["right hemidiaphragm"] = "Pleural"  # probably
    kg_dict["left lung"] = "Lung"
    kg_dict["left upper lung zone"] = "Lung"
    kg_dict["left mid lung zone"] = "Lung"
    kg_dict["left lower lung zone"] = "Lung"
    kg_dict["left hilar structures"] = "Lung"
    kg_dict["left apical zone"] = "Lung"
    kg_dict["left costophrenic angle"] = "Pleural"
    kg_dict["left hemidiaphragm"] = "Pleural"  # probably

    kg_dict["trachea"] = "Lung"
    kg_dict["right clavicle"] = "Bone"
    kg_dict["left clavicle"] = "Bone"
    kg_dict["aortic arch"] = "Heart"
    kg_dict["upper mediastinum"] = "Mediastinum"
    kg_dict["svc"] = "Heart"
    kg_dict["cardiac silhouette"] = "Heart"
    kg_dict["cavoatrial junction"] = "Heart"
    kg_dict["right atrium"] = "Heart"
    kg_dict["carina"] = "Lung"
    kg_dict["Edema"] = "Lung"

    return kg_dict


def get_vindr_label2id():
    dict = {}
    dict["Aortic enlargement"] = 0
    dict["Atelectasis"] = 1
    dict["Cardiomegaly"] = 2
    dict["Calcification"] = 3
    dict["Clavicle fracture"] = 4
    dict["Consolidation"] = 5
    dict["Edema"] = 6
    dict["Emphysema"] = 7
    dict["Enlarged PA"] = 8
    dict["ILD"] = 9
    dict["Infiltration"] = 10
    dict["Lung cavity"] = 11
    dict["Lung cyst"] = 12
    dict["Lung Opacity"] = 13
    dict["Mediastinal shift"] = 14
    dict["Nodule/Mass"] = 15
    dict["Pulmonary fibrosis"] = 16
    dict["Pneumothorax"] = 17
    dict["Pleural thickening"] = 18
    dict["Pleural effusion"] = 19
    dict["Rib fracture"] = 20
    dict["Other lesion"] = 21
    # dict["No finding"] = 22
    return dict


def get_kg():
    # anatomical part
    kg_dict = get_kg_ana_only()

    # disease part
    kg_dict["Aortic enlargement"] = "Heart"
    kg_dict["Atelectasis"] = "Lung"
    kg_dict["Calcification"] = "Bone"
    kg_dict["Cardiomegaly"] = "Heart"
    kg_dict["Consolidation"] = "Lung"
    kg_dict["ILD"] = "Lung"
    kg_dict["Infiltration"] = "Lung"
    kg_dict["Lung Opacity"] = "Lung"
    kg_dict["Nodule/Mass"] = "Lung"
    kg_dict["Other lesion"] = "Lung"
    kg_dict["Pleural effusion"] = "Pleural"
    kg_dict["Pleural thickening"] = "Pleural"
    kg_dict["Pneumothorax"] = "Pleural"
    kg_dict["Pulmonary fibrosis"] = "Lung"
    kg_dict["Clavicle fracture"] = "Bone"
    kg_dict["Emphysema"] = "Lung"
    kg_dict["Enlarged PA"] = "Heart"
    kg_dict["Lung cavity"] = "Lung"
    kg_dict["Lung cyst"] = "Lung"
    kg_dict["Mediastinal shift"] = "Mediastinum"
    kg_dict["Rib fracture"] = "Bone"
    kg_dict["Fracture"] = "Bone"

    return kg_dict


def get_countingAdj_name2index(feature_extraction_path):
    path = os.path.join(feature_extraction_path, "mimic-cxr-2.0.0-chexpert.csv.gz")
    df = pd.read_csv(path)

    mimic_list = df.columns[2:16].values

    ans2label = {key.lower(): i for i, key in enumerate(mimic_list)}
    return ans2label


def get_semantic_adj(pred_classes_ana_list, pred_classes_loc_list, feature_extraction_path):
    ana_thing_classes = list(get_kg_ana_only())
    ana_thing_classes = [ana_thing_classes[i].lower() for i in range(len(ana_thing_classes))]
    di_thing_classes = list(get_vindr_label2id())
    di_thing_classes = [di_thing_classes[i].lower() for i in range(len(di_thing_classes))]
    kg_ana = get_kg()  # ana_KG
    new_kg = {}
    for key in kg_ana:
        new_kg[key.lower()] = kg_ana[key]
    kg_ana = new_kg
    with open(os.path.join(feature_extraction_path, "GT_counting_adj.pkl"), "rb") as tf:
        small_counting_adj = pickle.load(tf)
        for i in range(len(small_counting_adj)):
            small_counting_adj[i] = small_counting_adj[i] / small_counting_adj[i][i]
        small_adj = np.where(small_counting_adj > 0.18, 2, 0)  # co-occurrence kg
    small_name2index = get_countingAdj_name2index(feature_extraction_path)

    (
        B,
        _,
    ) = pred_classes_ana_list.shape
    # pred_classes_di += len(ana_thing_classes)
    adj_matrix_list = []
    for element in range(B):
        pred_classes_ana = pred_classes_ana_list[element]
        pred_classes_loc = pred_classes_loc_list[element]
        pred_classes_loc += len(ana_thing_classes)

        # for i in range(len(di_thing_classes)):
        #     # di_thing_classes[i] = di_thing_classes[i].lower().replace(' ', '_')
        #     if 'fracture' in di_thing_classes[i]:
        #         di_thing_classes[i] = 'fracture'

        thing_classes = ana_thing_classes + di_thing_classes
        pred_classes = np.hstack((pred_classes_ana, pred_classes_loc))

        ana_thing_classes_set = set(ana_thing_classes)
        di_thing_classes_set = set(di_thing_classes)
        adj_matrix = np.zeros([100, 100], int)
        for i in range(52):
            for j in range(i, 52):
                if pred_classes[i] == len(thing_classes) or pred_classes[j] == len(thing_classes):
                    continue
                if kg_ana[thing_classes[pred_classes[i]]] == kg_ana[thing_classes[pred_classes[j]]]:
                    if (
                        thing_classes[pred_classes[i]] in ana_thing_classes_set
                        and thing_classes[pred_classes[j]] in di_thing_classes_set
                        or thing_classes[pred_classes[j]] in ana_thing_classes_set
                        and thing_classes[pred_classes[i]] in di_thing_classes_set
                    ):
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1

                # semantic garph 2
                if (
                    thing_classes[pred_classes[i]].lower() in small_name2index
                    and thing_classes[pred_classes[j]].lower() in small_name2index
                ):
                    # small_adj is the 14x14 co-occurrence matrix
                    value = max(
                        small_adj[
                            small_name2index[thing_classes[pred_classes[i]].lower()],
                            small_name2index[thing_classes[pred_classes[j]].lower()],
                        ],
                        adj_matrix[i, j],
                    )
                    adj_matrix[i, j] = value
                    adj_matrix[j, i] = value
        adj_matrix_list.append(adj_matrix)
    return np.array(adj_matrix_list)

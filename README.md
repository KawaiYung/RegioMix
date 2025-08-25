# Region-Specific Retrieval Augmentation for Longitudinal Visual Question Answering: A Mix-and-Match Paradigm

---

##  Prerequisites

This repository builds heavily on the prior work **EKAID** ([GitHub link](https://github.com/Holipori/EKAID/tree/main)).  
For detailed instructions on data preparation, please refer to the  [feature extraction guide](https://github.com/Holipori/EKAID/tree/main/feature%20extraction)

The cxr-record-list.csv can be found in the original [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.1.0/) 


---

##  Training Pipeline  

### **Step 1: Pre-compute Retrieval Indices**  
Generate retrieval indices for each sample:
```bash
python retrive_idx.py \
  --data-root <path_to_data> \
  --k <num_retrieved_objects> \
  --gpus <num_gpus>
```

---

### **Step 2: Split Pre-computed Features**  
Align pre-computed features with the training split only:
```bash
python split_cmb_features.py \
  --data-root <path_to_data>
```

---

### **Step 3: Split Disease Reports**  
Align the reports with the training split only. The cxr-record-list.csv can be found in the original [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.1.0/):
```bash
python split_report_disease.py \
  --data-root <path_to_data>
```

---

### **Step 4: Preprocess Retrieval Features**  
Group retrieved features for efficient model input:
```bash
python retrieval_preprocessing.py \
  --data-root <path_to_data> \
  --k <num_retrieved_objects>
```

---

### **Step 5: Train the Model**  
Run the training script with retrieval augmentation:
```bash
python train_mimic.py \
  --graph all \
  --eval_target test \
  --eval_difference \
  --lr 0.00005 \
  --reaug_k 5 \
  --seed 1250 \
  --infonce 0.5 \
  --use_allign
```

---


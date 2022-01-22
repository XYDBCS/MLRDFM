from mine_function import *
from sklearn.model_selection import KFold
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import config
sys.path.append("..")
from DeepFM import DeepFM
A = np.load("./data/HMDD3.2/miRNA-disease association.npy")
x,y = A.shape
samples = get_balance_samples(A)#return array, including the same number of positive samples and negative samples
label_all = []
y_score_all = []
fold_num = 5
embedding_n = 30
#cross validation
kf = KFold(n_splits=fold_num, shuffle=True)
iter = 0 #control each iterator
sum_score = 0
# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": embedding_n,
    "dropout_fm": [1, 1],
    "deep_layers": [64, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch":180,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 0,
    "batch_norm_decay": 0.995,
    "verbose": True,
    "l2_reg": 0.01,
    "random_seed": config.RANDOM_SEED,
}

roc_sum = 0
AUPR_sum = 0
k =2
th = 0 #contral to get the similarity associations

dfm_params["feature_size"] = (x + y)
dfm_params["field_size"] = 2

AUC_all = []
AUPR_all = []
for train_index, test_index in kf.split(samples):
    if iter < 6:
        iter = iter + 1
        train_samples = samples[train_index, :]
        test_samples = samples[test_index, :]
        new_A = update_Adjacency_matrix(A, test_samples)
        sim_m, sim_d = get_syn_sim(new_A,x, y )
        dfm_params["sim_m"] = sim_m
        dfm_params["sim_d"] = sim_d
        train_fea, y_train_, test_fea, test_y = get_feature_label(new_A, train_samples, test_samples,k)
        Xi_train_, Xv_train_, Xi_valid_, Xv_valid_ = data_transform(train_fea, test_fea, k)
        y_valid_ = array_list(test_y)
        dfm = DeepFM(**dfm_params)
        AUC_list, AUPR_list = dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        AUC_all.append(AUC_list)
        AUPR_all.append(AUPR_list)
        pre = []
        pre = dfm.predict(Xi_valid_, Xv_valid_)
        fpr, tpr, thersholds = roc_curve(test_y, pre.reshape(-1), pos_label=1)
        label_all.extend(test_y)
        y_score_all.extend(pre.reshape(-1))
        roc_auc = auc(fpr, tpr)
        precision2, recall2, _thresholds2 = precision_recall_curve(test_y, pre.reshape(-1))
        AUPR = auc(recall2, precision2)
        AUPR_sum = AUPR_sum + AUPR
        print("AUC", iter, roc_auc)
        print("AUPR", iter, AUPR)
        roc_sum = roc_sum+roc_auc
print("average_roc", roc_sum/fold_num, "average_AUPR", AUPR_sum/fold_num)

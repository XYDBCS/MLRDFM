# MLRDFM
This project includes a Tensorflow implementation of a Multi-view Laplacian regularized DeepFM model for miRNA-disease association prediction
 
The code is for the Paper "MLRDFM: a multi-view Laplacian regularized DeepFM model for predicting miRNA-disease associations".
If you use this code, please cite this paper.

# Usage
## Input Format
This implementation requires the input data in the following format:
- [ ] **Xi**: *[[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]*
    - *indi_j* is the feature index of feature field *j* of sample *i* in the dataset
- [ ] **Xv**: *[[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]*
    - *vali_j* is the feature value of feature field *j* of sample *i* in the dataset
    - *vali_j* can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
- [ ] **y**: target of each sample in the dataset (1/0 for classification, numeric number for regression)

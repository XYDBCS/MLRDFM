import numpy as np
import random
from numpy import linalg
from scipy.linalg import eigh
def read_txt1(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split( )
            row = []
            for k in line:
                row.append(int(float(k)))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data
def get_balance_samples(A): # return the same number of negative samples and postive samples, and all the negative samples
    m,n = A.shape
    pos = []
    neg = []
    temp_neg_row = []
    for i in range(m):
        pos_row_n = 0
        for j in range(n):
            if A[i,j] ==1:
                pos_row_n = pos_row_n+1
                pos.append([i,j,1])
            else:
                temp_neg_row.append([i,j,0])
        neg_row = random.sample(temp_neg_row, pos_row_n)
        neg = neg + neg_row
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    np.random.shuffle(samples)
    return samples


def update_Adjacency_matrix (A, test_samples):
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i,2] ==1:
            #print("index", test_samples[i,0], test_samples[i,1] )
            A_tep [int(test_samples[i,0]), int(test_samples[i,1])] = 0
    return A_tep


def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix
def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def get_feature_label(new_matrix, train_samples, test_samples, k):
    m= train_samples.shape[0]
    n= test_samples.shape[0]
    print("m,n", m, n)
    h, v = new_matrix.shape

    new_matrix_T = new_matrix.transpose()
    if k==1:
        train_fea = np.zeros([m, h+v])
        train_y = np.zeros([m])
        test_fea = np.zeros([n, h+v])
        test_y = np.zeros([n])
        for i in range(m):
            train_fea[i,:] =np.hstack((new_matrix[train_samples[i,0], :], new_matrix_T[train_samples[i, 1], :]))
            train_y[i] = train_samples[i,2]
        for i in range(n):
            test_fea[i,:] = np.hstack((new_matrix[test_samples[i,0], :], new_matrix_T[test_samples[i, 1],:]))
            test_y[i] = test_samples[i,2]
    elif k ==0:
        k1 = 50
        k2 = 37
        sim_m, sim_d = get_syn_sim(new_matrix, k1, k2)
        print("the maxmum of sim network", np.max(np.max(sim_m, axis=0)), "the minimum of sim network",
              np.min(np.min(sim_m, axis=0)))
        print("the maxmum of simd network", np.max(np.max(sim_d, axis=0)), "the minimum of simd network",
              np.min(np.min(sim_d, axis=0)))
        m_m = change_sim_to_associa(sim_m, 1)  # miRNA_miRNA association
        d_d = change_sim_to_associa(sim_d, 1)  # disease_disease association
        train_fea = np.zeros([m, 2*(h + v)])
        train_y = np.zeros([m])
        test_fea = np.zeros([n, 2*(h + v)])
        test_y = np.zeros([n])
        for i in range(m):
            temp1 = np.hstack((new_matrix[train_samples[i, 0], :], new_matrix_T[train_samples[i, 1], :]))
            temp = np.hstack((m_m[train_samples[i,0],:], d_d[train_samples[i,1], :]))
            train_fea[i, :] = np.hstack((temp1, temp))
            train_y[i] = train_samples[i, 2]
        print("train_fea shape", train_fea.shape)
        for i in range(n):
            temp_t1 = np.hstack((new_matrix[test_samples[i, 0], :], new_matrix_T[test_samples[i, 1], :]))
            temp_t = np.hstack((m_m[test_samples[i, 0], :], d_d[test_samples[i, 1], :]))
            test_fea[i, :] = np.hstack((temp_t1, temp_t))
            test_y[i] = test_samples[i, 2]

    elif k==2:
        train_fea = np.zeros([m, 2])
        train_y = np.zeros([m])
        test_fea = np.zeros([n, 2])
        test_y = np.zeros([n])
        for i in range(m):
            train_fea[i, 0] = train_samples[i, 0]
            train_fea[i, 1] = h + train_samples[i, 1]
            train_y[i] = train_samples[i, 2]
        for i in range(n):
            test_fea[i, 0] = test_samples[i, 0]
            test_fea[i, 1] = h + test_samples[i, 1]
            test_y[i] = test_samples[i, 2]
    elif k==3:
        k1 = 50
        k2 = 37
        sim_m, sim_d = get_syn_sim(new_matrix, k1, k2)
        train_fea = np.zeros([m, 2 + (h + v)])
        train_y = np.zeros([m])
        test_fea = np.zeros([n, 2 + (h + v)])
        test_y = np.zeros([n])
        for i in range(m):
            train_fea[i, 0] = train_samples[i, 0]
            train_fea[i, 1] = h + train_samples[i, 1]
            train_y[i] = train_samples[i, 2]
        for i in range(n):
            test_fea[i, 0] = test_samples[i, 0]
            test_fea[i, 1] = h + test_samples[i, 1]
            test_y[i] = test_samples[i, 2]

        train_fea[i, 2:(2 + (h + v))] = np.hstack((sim_m[train_samples[i, 0], :],
                                                   sim_d[train_samples[i, 1], :]))

        train_fea[i, 2:(2 + (h + v))] = np.hstack((sim_m[test_samples[i, 0], :],
                                                   sim_d[test_samples[i, 1], :]))

    elif k==4: #the combination of 1 and 2
        train_fea = np.zeros([m, 2+h + v])
        train_y = np.zeros([m])
        test_fea = np.zeros([n, 2+h + v])
        test_y = np.zeros([n])
        for i in range(m):
            train_fea[i, 0] = train_samples[i, 0]
            train_fea[i, 1] = h + train_samples[i, 1]
            train_fea[i, 2:(2+h+v)] = np.hstack((new_matrix[train_samples[i, 0], :], new_matrix_T[train_samples[i, 1], :]))
            train_y[i] = train_samples[i, 2]
        for i in range(n):
            test_fea[i, 0] = test_samples[i, 0]
            test_fea[i, 1] = h + test_samples[i, 1]
            test_fea[i, 2:(2+h+v)] = np.hstack((new_matrix[test_samples[i, 0], :], new_matrix_T[test_samples[i, 1], :]))
            test_y[i] = test_samples[i, 2]
    y_train_ = array_list(train_y)

    return train_fea, y_train_, test_fea, test_y


def get_feature_label1(new_matrix, train_samples, test_samples, mic_lnc, lnc_dis, k):

    m= train_samples.shape[0]
    n= test_samples.shape[0]
    print("m,n", m, n)
    h, v = new_matrix.shape
    new_matrix_T = new_matrix.transpose()
    z = lnc_dis.shape[0]
    train_y = np.zeros([m])
    test_y = np.zeros([n])
    if k ==5:
        train_fea = np.zeros([m, 2 + h + v + z])
        test_fea = np.zeros([n, 2 + h + v + z])
        for i in range(m):
            train_fea[i, 0] = train_samples[i, 0]
            train_fea[i, 1] = h + train_samples[i, 1]
            train_fea[i, 2:(2+h+v)] = np.hstack((new_matrix[train_samples[i, 0], :], new_matrix_T[train_samples[i, 1], :]))
            mic_c = mic_lnc[train_samples[i, 0],:]
            dis_c = lnc_dis[:,train_samples[i, 1]]
            #1 is mic=0 and dis =0, 2 is mic = 1, dis =0, 3 is mic =0, dis = 1, 4 is mic=1, dis = 1
            lnc_ass = np.zeros([z])
            for j in range(z):
                if mic_c[j] == 0 and dis_c[j] ==0:
                    lnc_ass[j] =1
                elif mic_c[j] ==1 and dis_c[j] ==0:
                    lnc_ass[j] =2
                elif mic_c[j] ==0 and dis_c[j] ==1:
                    lnc_ass[j] == 3
                elif mic_c[j] ==1 and dis_c[j] ==1:
                    lnc_ass[j] =4
            train_fea[i, (2 + h + v):(2+h+v+z)] = lnc_ass
            train_y[i] = train_samples[i, 2]
        for i in range(n):
            test_fea[i, 0] = test_samples[i, 0]
            test_fea[i, 1] = h + test_samples[i, 1]
            test_fea[i, 2:(2+h+v)] = np.hstack((new_matrix[test_samples[i, 0], :], new_matrix_T[test_samples[i, 1], :]))
            mic_c = mic_lnc[test_samples[i, 0],:]
            dis_c = lnc_dis[:,test_samples[i, 1]]
            lnc_ass = np.zeros([z])
            for j in range(z):
                if mic_c[j] ==0 and dis_c[j] ==0:
                    lnc_ass[j] =1
                elif mic_c[j] ==1 and dis_c[j] ==0:
                    lnc_ass[j] =2
                elif mic_c[j] ==0 and dis_c[j] ==1:
                    lnc_ass[j] = 3
                elif mic_c[j] ==1 and dis_c[j] ==1:
                    lnc_ass[j] =4
            test_fea[i, (2 + h + v):(2+h+v+z)] = lnc_ass
            test_y[i] = test_samples[i, 2]
    elif k ==6:
        for i in range(m):
            train_fea = np.zeros([m, 2 + h + v + 2*z])
            test_fea = np.zeros([n, 2 + h + v + 2*z])
            train_fea[i, 0] = train_samples[i, 0]
            train_fea[i, 1] = h + train_samples[i, 1]
            train_fea[i, 2:(2 + h + v)] = np.hstack(
                (new_matrix[train_samples[i, 0], :], new_matrix_T[train_samples[i, 1], :]))
            mic_c = mic_lnc[train_samples[i, 0], :]
            dis_c = lnc_dis[:, train_samples[i, 1]]
            dis_c1 = dis_c.transpose()
            # 1 is mic=0 and dis =0, 2 is mic = 1, dis =0, 3 is mic =0, dis = 1, 4 is mic=1, dis = 1
            train_fea[i, (2 + h + v):(2 + h + v + 2*z)] = np.hstack((mic_c, dis_c1))
            train_y[i] = train_samples[i, 2]
        for i in range(n):
            test_fea[i, 0] = test_samples[i, 0]
            test_fea[i, 1] = h + test_samples[i, 1]
            test_fea[i, 2:(2 + h + v)] = np.hstack(
                (new_matrix[test_samples[i, 0], :], new_matrix_T[test_samples[i, 1], :]))
            mic_c = mic_lnc[test_samples[i, 0], :]
            dis_c = lnc_dis[:, test_samples[i, 1]]
            dis_c1 = dis_c.transpose()
            # 1 is mic=0 and dis =0, 2 is mic = 1, dis =0, 3 is mic =0, dis = 1, 4 is mic=1, dis = 1
            test_fea[i, (2 + h + v):(2 + h + v + 2*z)] = np.hstack((mic_c, dis_c1))
            test_y[i] = test_samples[i, 2]
    return train_fea, train_y, test_fea, test_y

def data_transform (train_fea, test_fea, k):#k as the same meaning of the upper function
    m,n = train_fea.shape
    m1, n1 = test_fea.shape
    print("n, n1", n, n1)
#for 01 as the feature of association with miRNA1
    if k ==1:
        train_xi = np.zeros([m,n])
        train_xv = np.ones([m,n])
        test_xi = np.zeros([m1,n1])
        test_xv = np.ones([m1,n1])
        for i in range(m):
            for j in range(n):
                train_xi[i,j] = 2*j+train_fea[i,j]
        for i in range(m1):
            for j in range(n1):
                test_xi[i,j] = 2*j+test_fea[i,j]
    elif k ==0:
        # get the miRNA-miRNA asso and disease-disease ass from similarity network
        n = int(n/2)
        n1 = int(n1/2)
        train_xi = np.zeros([m, n])
        train_xv = np.ones([m, n])
        test_xi = np.zeros([m1, n1])
        test_xv = np.ones([m1, n1])
        for i in range(m):
            for j in range(n):
                    if train_fea[i, j] ==0 and train_fea[i, j+ n] ==0:
                        train_xi[i,j] = 4*j
                    elif train_fea[i,j] ==1 and train_fea[i, j+n] ==0:
                        train_xi[i,j] = 4*j +1
                    elif train_fea[i,j] ==0 and train_fea[i, j+n] ==1:
                        train_xi[i,j] = 4*j +2
                    elif train_fea[i,j] ==1 and train_fea[i,j+n] ==1:
                        train_xi[i,j] = 4*j+3
        for i in range(m1):
            for j in range(n1):
                if test_fea[i, j] == 0 and test_fea[i, j + n] == 0:
                    test_xi[i, j] = 4 * j
                elif test_fea[i, j] == 1 and test_fea[i, j + n] == 0:
                    test_xi[i, j] = 4 * j + 1
                elif test_fea[i, j] == 0 and test_fea[i, j + n] == 1:
                    test_xi[i, j] = 4 * j + 2
                elif test_fea[i, j] == 1 and test_fea[i, j + n] == 1:
                    test_xi[i, j] = 4 * j + 3

    elif k==2: #only us a onehot miRNA and a one hot disease
        train_xi = np.zeros([m, 2])
        train_xv = np.ones([m, 2])
        test_xi = np.zeros([m1, 2])
        test_xv = np.ones([m1, 2])
        for i in range(m):
            for j in range(2):
                train_xi[i,j] = train_fea[i,j]
        for i in range(m1):
            for j in range(2):
                test_xi[i,j] = test_fea[i,j]
    elif k==3:
        train_xi = np.zeros([m, n])
        train_xv = train_fea
        test_xi = np.zeros([m1, n])
        test_xv = test_fea
        field2_l = max(max(train_fea[:, 1]), max(test_fea[:, 1]))
        for i in range(m):
            for j in range(n):
                if j<2:
                    train_xi[i, j] = train_fea[i, j]
                    train_xv[i,j] = 1
                else:
                    train_xi[i, j] = field2_l + j - 1

        for i in range(m1):
            for j in range(n):
                if j<2:
                    test_xi[i, j] = test_fea[i, j]
                    test_xv[i,j] =1
                else:
                    test_xi[i, j] = field2_l + j - 1
    elif k==4:
        train_xi = np.zeros([m, n])
        train_xv = np.ones([m, n])
        test_xi = np.zeros([m1, n1])
        test_xv = np.ones([m1, n1])
        for i in range(m):
            for j in range(n):
                if j <2:
                    train_xi[i,j] = train_fea[i, j]
                else:
                    train_xi[i, j] =n-2 + 2 * (j-2) + train_fea[i, j]
        for i in range(m1):
            for j in range(n1):
                if j<2:
                    test_xi[i, j] = test_fea[i, j]
                else:
                    test_xi[i, j] = n-2 + 2 * (j-2) + test_fea[i, j]

    Xi_train_ = array_list(train_xi)
    Xv_train_ = array_list(train_xv)
    Xi_valid_ = array_list(test_xi)
    Xv_valid_ = array_list(test_xv)
    return Xi_train_, Xv_train_, Xi_valid_, Xv_valid_

def data_transform1 (train_fea, test_fea, l_n, k ):#k as the same meaning of the upper function
    m, n = train_fea.shape
    m1, n1 = test_fea.shape
    train_xi = np.zeros([m, n])
    train_xv = np.ones([m, n])
    test_xi = np.zeros([m1, n1])
    test_xv = np.ones([m1, n1])
    if k ==5:
        for i in range(m):
            for j in range(n):
                if j < 2:
                    train_xi[i, j] = train_fea[i, j]
                elif 2<=j< n-l_n:
                    train_xi[i, j] = n - 2-l_n + 2 * (j - 2) + train_fea[i, j]
                else:
                    train_xi[i,j] = 3*(n-2-l_n)+4*(j - (n-l_n)) + train_fea[i, j]

        for i in range(m1):
            for j in range(n1):
                if j < 2:
                    test_xi[i, j] = test_fea[i, j]
                elif 2 <= j < n - l_n:
                    test_xi[i, j] = n - 2 - l_n + 2 * (j - 2) + test_fea[i, j]
                else:
                    test_xi[i, j] = 3 * (n - 2 - l_n) + 4 * (j - (n - l_n)) + test_fea[i, j]-1
    elif k ==6:
        for i in range(m):
            for j in range(n):
                if j < 2:
                    train_xi[i, j] = train_fea[i, j]
                else:
                    train_xi[i, j] = n - 2 - 2*l_n + 2 * (j - 2) + train_fea[i, j]

        for i in range(m1):
            for j in range(n1):
                if j < 2:
                    test_xi[i, j] = test_fea[i, j]
                else:
                    test_xi[i, j] = n - 2 - 2*l_n + 2 * (j - 2) + test_fea[i, j]

    return train_xi, train_xv, test_xi, test_xv


def array_list(arry):
    li = arry.tolist()
    return li

def get_lapl_matrix(sim):
    m,n = sim.shape
    lap_matrix_tep = np.zeros([m,m])
    for i in range(m):
        lap_matrix_tep[i,i] = np.sum(sim[i,:])
    lap_matrix = lap_matrix_tep - sim
    return lap_matrix, lap_matrix_tep

def get_initial_weights_by_manifold(k, sim_m, sim_d):

    m,m = sim_m.shape
    n,n = sim_d.shape
    lap_matx_m, lap_Dm = get_lapl_matrix(sim_m)
    lap_matx_d, lap_Dd = get_lapl_matrix(sim_d)
    # vu_m, vec_m = np.linalg.eig(lap_matx_m)# vu_m is the eig_value, and vec_m is the eig_vector of miRNAs
    # vu_d, vec_d = np.linalg.eig(lap_matx_d)
    vu_m, vec_m = eigh(lap_matx_m, lap_Dm)
    vu_d, vec_d = eigh(lap_matx_d, lap_Dd)
    vu_m_ind = np.argsort(vu_m)
    vu_d_ind = np.argsort(vu_d)
    m_embedding_w = np.zeros([m, k])
    v_embedding_w = np.zeros([n, k])
    for i in range(k):
        m_embedding_w[:, i] = vec_m[vu_m_ind[i+1], :]
        v_embedding_w[:, i] = vec_d[vu_d_ind[i+1], :]
    embedding_weight = np.vstack((m_embedding_w, v_embedding_w))
    return embedding_weight



#W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn


#updataing rules
def MiRNA_updating (S1,S2,S3,S4, P1,P2,P3,P4):
    it = 0
    P = (P1+P2+P3+P4)/4
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2+P3+P4)/3),S1.T)
        P111 = new_normalization(P111)
        # P222 =np.dot (np.dot(S2,(P1+P3+P4)/3),S2.T)
        # P222 = new_normalization(P222)
        # P333 = np.dot (np.dot(S3,(P1+P2+P4)/3),S3.T)
        # P333 = new_normalization(P333)
        P444 = np.dot(np.dot(S4,(P1+P2+P3)/3),S4.T)
        P444 = new_normalization(P444)
        P1 = P111
        # P2 = P222
        # P3 = P333
        P4 = P444
        # P_New = (P1+P2+P3+P4)/4
        P_New = (P1 + P4) / 4
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def disease_updating(S1,S2, P1,P2):
    it = 0
    P = (P1+P2)/2
    dif = 1
    while dif> 0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,P2),S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot (np.dot(S2,P1),S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1+P2)/2
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P

def get_syn_sim (A, k1, k2):
    disease_sim1 = read_txt1("./data/HMDD3.2/disease_semantic_sim.txt")
    miRNA_sim1 = read_txt1("./data/HMDD3.2/miRNA_functional_sim.txt")
    miRNA_sim2 = read_txt1 ("./data/HMDD3.2/miRNA_sequence_sim.txt")
    miRNA_sim3 = read_txt1("./data/HMDD3.2/miRNA_semantic_sim.txt")
    GIP_m_sim = (GIP_kernel(A)+miRNA_sim1+miRNA_sim2+miRNA_sim3)/4
    GIP_d_sim= (GIP_kernel(A.T)+disease_sim1)/2
    for i in range(k1):
        GIP_m_sim[i,i] = 0
    for j in range(k2):
        GIP_d_sim[j,j] = 0
    Pm_final = GIP_m_sim
    Pd_final = GIP_d_sim
    return Pm_final, Pd_final





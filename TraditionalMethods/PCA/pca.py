import cv2 as cv
import numpy as np
import pickle
import pdb

# train data
def pca_compress(data_mat, k=9999999):
    '''
    :param data_mat: 输入数据
    :param k: output dimension
    :return: low_dim_data comporessed matrix, 
    '''
    # 1. 数据中心化
    X = np.array(data_mat)
    mean_vals = X.mean(axis=0)
    X = X - mean_vals
    # 2. 计算协方差矩阵
    CX = np.cov(X.T)
    # 3. 计算特征值和特征向量
    eigenvalue, featurevector = np.linalg.eig(CX)
    # print(eigenvalue)
    # 4. 选取出K个最大到特征值所对应的特征向量
    mat = np.concatenate(
        (eigenvalue.reshape(-1, 1), featurevector.T),
        axis=1
    )
    mat_sorted = mat[np.argsort(mat[:,0])[::-1]]
    re_eig_vects = mat_sorted[:k, 1:]
    print("re_eig_vects shape: ", re_eig_vects.shape)
    P = re_eig_vects.T
    # 5. 计算投影后的数据
    low_dim_data = X @ P
    print("low_dim_data shape: ", low_dim_data.shape)
    print("mean_vals shape: ", mean_vals.shape)
    return low_dim_data, mean_vals, re_eig_vects


# test data
def test_img(img, mean_vals, re_eig_vects):
    mean_removed = img.reshape(1, -1) - mean_vals
    # (1 x 10304) * (120 x 360) ?
    return mean_removed @ re_eig_vects.T


# compute the distance between vectors using euclidean distance
def compute_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1)[0] - np.array(vector2)[0])


# compute the distance between vectors using cosine distance
def compute_distance_(vector1, vector2):
    vector1 = vector1.reshape(1, -1)
    # print("vect1 shape", np.array(vector1)[0])
    # print("vect2 shape", np.array(vector2)[0])
    # pdb.set_trace()
    return np.dot(np.array(vector1)[0], np.array(vector2)[0]) / (np.linalg.norm(np.array(vector1)[0]) * (np.linalg.norm(np.array(vector2)[0])))

def save(low_dim_data, mean_vals, re_eig_vects):
    output = {
        "low_dim_data": low_dim_data,
        "mean_vals": mean_vals,
        "re_eig_vects": re_eig_vects
    }
    pickle.dump(output, open('./cache_data.dat', 'wb'))


def load():
    output = pickle.load(open('./cache_data.dat', 'rb'))
    return output['low_dim_data'], output['mean_vals'], output['re_eig_vects']

if __name__ == '__main__':

    # 1. use num 1- 9 image of each person to train
    data = []
    for i in range(1, 41):
        for j in range(1, 10):
            img = cv.imread('orl_faces/s' + str(i) + '/' + str(j) + '.pgm', 0)
            width, height = img.shape
            img = img.reshape((img.shape[0] * img.shape[1]))
            data.append(img)

    # low_dim_data, mean_vals, re_eig_vects = pca_compress(data, 120)

    # save(low_dim_data, mean_vals, re_eig_vects)
    low_dim_data, mean_vals, re_eig_vects = load()


    # 2. use num 10 image of each person to test
    correct = 0
    for k in range(1, 41):
        img = cv.imread('orl_faces/s' + str(k) + '/10.pgm', 0)
        img = img.reshape((img.shape[0] * img.shape[1]))
        distance = test_img(img, mean_vals, re_eig_vects)
        distance_mat = []
        for i in range(1, 41):
            for j in range(1, 10):
                distance_mat.append(compute_distance_(low_dim_data[(i - 1) * 9 + j - 1], distance.reshape((1, -1))))
        distance_mat = np.array(distance_mat)
        # print("distance mat len: ", distance_mat.shape)
        num_ = np.argmax(distance_mat)
        # print(num_)
        class_ = int(num_ / 9) + 1
        if class_ == k:
            correct += 1
        print('s' + str(k) + '/10.pgm is the most similar to s' +
              str(class_) + '/' + str(num_ % 9 + 1) + '.pgm')
    print("accuracy: %lf" % (correct / 40))

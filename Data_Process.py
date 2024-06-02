import numpy as np
import pandas as pd
import os
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler


def read_img(path, categories, img_size, channel):
    X = []
    y = []
    for target, category in enumerate(categories):
        p = os.path.join(path, category)
        for img_file in os.listdir(p):
            img = cv.imread(os.path.join(p, img_file), 1)
            if channel == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (img_size, img_size))
            X.append(img)
            y.append(target)
    X = np.array(X).reshape(-1, img_size, img_size, channel)
    X = X / 255.0
    return X, np.array(y)


def read_dataset(data, path, img_size, channel):
    if data <= 2:
        if data == 1:
            categories = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
            val_idx = [132, 260, 7, 8, 266, 139, 15, 144, 146, 147, 20, 278, 23, 164, 40, 169, 297, 175, 307, 310, 184,
                       59, 322, 73, 75, 343, 87, 88, 346, 93, 225, 113, 126]
        else:
            categories = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
            val_idx = [54, 44, 98, 26, 27, 90, 70, 148, 71, 82, 94, 156, 210, 176, 162, 167, 181, 182, 202, 206, 169,
                       204, 172, 177, 155, 316, 339, 222, 219, 325, 304]

        dataset = np.array(pd.read_csv(path + '.csv', header=0))

        dsize = dataset.shape[0]
        k = len(categories)

        tmp = dataset.T[:-1].T
        if data == 2:
            island = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
            sex = {np.nan: 0, 'MALE': 1, 'FEMALE': 2}
            for i in range(dsize):
                tmp[i][0] = island[tmp[i][0]]
                tmp[i][5] = sex[tmp[i][5]]

        samples = np.array([[float(num) for num in row] for row in tmp])
        samples = MinMaxScaler().fit_transform(samples)

        targets = dataset.T[-1]
        for i in range(dsize):
            targets[i] = categories[targets[i]]

        # c = []
        # var = []
        # for i in range(k):
        #     x = [index for index, element in enumerate(targets) if element == i]
        #     c.append(len(x))
        #     v = np.var(samples[x], axis=0)
        #     v = np.mean(v)
        #     var.append(v)
        # c = c/np.max(c)
        # print(np.var(c))
        # x = []
        # for i in range(k):
        #     x.append(c[i]*var[i])
        # print(x)
        # exit(0)

        train_idx = list(set(range(dsize)) - set(val_idx))
        X_train = samples[train_idx]
        X_val = samples[val_idx]
        y_train = targets[train_idx]
        y_val = targets[val_idx]

    else:
        if data == 3:
            categories = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
                          "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                          "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                          "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
        else:
            categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        X_train, y_train = read_img(path + 'train/', categories, img_size, channel)
        X_val, y_val = read_img(path + 'val/', categories, img_size, channel)
        k = 10

    return X_train, y_train, X_val, y_val, k

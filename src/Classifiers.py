import numpy as np
from keras import backend
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping


def softmax(scores):
    e_x = np.exp(scores)
    return e_x / np.sum(e_x, axis=0)


class SoftmaxRegression:

    def __init__(self, X_train, y_train, X_val, y_val, k):
        self.X_train = X_train
        self.y_train = to_categorical(y_train)
        self.X_val = X_val
        self.y_val = y_val
        self.k = k
        self.m = X_train.shape[1]
        self.val_size = X_val.shape[0]

    def acc_calc(self, probs):

        acc = 0.0
        for i in range(self.val_size):
            acc += np.argmax(probs[i]) == self.y_val[i]

        return acc / self.val_size

    def fit(self, labelled, lr=0.01, tol=1e-5, epochs=50000):

        L_size = len(labelled)
        X = self.X_train[labelled]
        y = self.y_train[labelled].T

        w = np.zeros([self.k, self.m])

        for e in range(epochs):
            probs = softmax(w.dot(X.T))
            grad = (-lr / L_size) * (y - probs).dot(X)
            w -= grad
            if (np.abs(grad) < tol).all():
                break

        probs = softmax(w.dot(self.X_val.T)).T
        acc = self.acc_calc(probs)
        probs = softmax(w.dot(self.X_train.T)).T

        return probs, acc


class CNN:

    def __init__(self, X_train, y_train, X_val, y_val, k, batch_size, img_size, channel):
        self.X_train = X_train
        self.y_train = to_categorical(y_train)
        self.X_val = X_val
        self.y_val = to_categorical(y_val)
        self.k = k

        self.batch_size = batch_size
        self.img_size = img_size
        self.channel = channel

    def net_init(self):

        classifier = Sequential()

        classifier.add(Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, self.channel), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.2))

        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.2))

        classifier.add(Conv2D(128, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.4))

        classifier.add(Flatten())

        classifier.add(Dense(activation='relu', units=64))
        classifier.add(Dense(activation='relu', units=128))
        classifier.add(Dense(activation='relu', units=64))
        classifier.add(Dense(activation='softmax', units=self.k))

        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # classifier.summary()

        return classifier

    def fit(self, labelled):

        classifier = self.net_init()

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0)
        classifier.fit(self.X_train[labelled], self.y_train[labelled], epochs=200, batch_size=self.batch_size,
                       callbacks=[early_stopping], verbose=0)

        probs = classifier.predict(self.X_train, verbose=0)
        acc = classifier.evaluate(self.X_val, self.y_val, verbose=0)[1]

        del classifier
        backend.clear_session()

        return probs, acc

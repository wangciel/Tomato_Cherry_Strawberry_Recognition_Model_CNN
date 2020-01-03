'''
Student Name: YiFan Wang
Student ID:300304266
'''
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from util.data_loader import load_mlp_train_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,train_test_split

seed = 309
random.seed(seed)
np.random.seed(seed)

data,labels = load_mlp_train_data()
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2,shuffle=True)

if __name__ == '__main__':

    kfold_scores = []

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 128, 32), activation='relu', learning_rate_init=0.001, max_iter=10000)
    kf = KFold(n_splits=4, shuffle=True)
    start_time = time.time()

    for train_index, test_index in kf.split(X_train):
        x_train, y_train, x_test, y_test = X_train[train_index], Y_train[train_index], data[test_index], labels[test_index]
        mlp.fit(x_train, y_train)
        y_predict = mlp.predict(x_test)

        kfold_scores.append(accuracy_score(y_test, y_predict))

    print("--- training time %s seconds ---" % (time.time() - start_time))

    plot = plt.figure()
    plt.plot(kfold_scores)
    plt.title('MLP_kfold_accuracy')
    plt.xlabel('kfold_batch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy'], loc='lower right')
    plot.savefig('./plots/MLP_kfold_accuracy.png')

    Y_predict = mlp.predict(X_test)
    total_accuracy = accuracy_score(Y_test,  Y_predict)
    print("accuracy={}".format(np.mean(total_accuracy)))









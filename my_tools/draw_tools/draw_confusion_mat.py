import itertools

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{:0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)

    # fig, ax = plt.subplots()
    # im = ax.imshow(cm)
    #
    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    # ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(vegetables)):
    #     for j in range(len(farmers)):
    #         text = ax.text(j, i, harvest[i, j],
    #                        ha="center", va="center", color="w")
    #
    # ax.set_title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()
    # plt.show()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, None)
    plt.yticks(tick_marks, None)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    # plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    '''
                             precision    recall  f1-score   support

      1 industrial land     0.6852    0.7400    0.7115       200
          10 shrub land     0.7708    0.7400    0.7551       200
   11 natural grassland     0.8762    0.9200    0.8976       200
12 artificial grassland     0.8476    0.6950    0.7637       200
               13 river     0.5891    0.5950    0.5920       200
                14 lake     0.6295    0.7900    0.7007       200
                15 pond     0.6339    0.3550    0.4551       200
    2 urban residential     0.6250    0.6750    0.6490       200
    3 rural residential     0.7421    0.7050    0.7231       200
         4 traffic land     0.7075    0.7500    0.7282       200
          5 paddy field     0.6781    0.7900    0.7298       200
       6 irrigated land     0.4089    0.4600    0.4329       200
         7 dry cropland     0.7838    0.7250    0.7532       200
          8 garden plot     0.7407    0.5000    0.5970       200
       9 arbor woodland     0.6732    0.8650    0.7571       200

               accuracy                         0.6870      3000
              macro avg     0.6928    0.6870    0.6831      3000
           weighted avg     0.6928    0.6870    0.6831      3000

[[148   0   0   0   0   0   0  32   7   9   0   0   3   1   0]
 [  0 148   6   4   0   0   0   0   5   2   3   1   0   4  27]
 [  0   1 184   1   0   0   1   0   0   0   8   5   0   0   0]
 [  0   9   0 139   0   0   0   0   0   4   3  34   0  10   1]
 [  8   0   0   0 119  43  10   1   0   0  13   4   0   0   2]
 [  1   0   0   0  14 158  21   0   0   0   2   2   1   0   1]
 [  1   0   0   0  52  48  71   0   0   0  14  10   0   1   3]
 [ 25   0   0   0   0   0   0 135  31   7   0   1   1   0   0]
 [ 13   1   0   0   0   0   0  39 141   4   0   0   1   1   0]
 [ 12   1   0   1   1   0   0   5   3 150   1  15   9   2   0]
 [  3   1   4   2   6   0   5   1   1   4 158  14   0   0   1]
 [  2   3  11  11   8   2   3   1   1  13  28  92  14   3   8]
 [  3   2   0   1   1   0   1   2   1  15   0  17 145   8   4]
 [  0  17   4   4   0   0   0   0   0   4   0  24  10 100  37]
 [  0   9   1   1   1   0   0   0   0   0   3   6   1   5 173]]
    '''
    cnf_matrix = np.load('confusion_matrix.npy')
    attack_types = ['1 industrial land', '10 shrub land', '11 natural grassland', '12 artificial grassland', '13 river', '14 lake', '15 pond', '2 urban residential', '3 rural residential', '4 traffic land', '5 paddy field', '6 irrigated land', '7 dry cropland', '8 garden plot', '9 arbor woodland']
    attack_types = [' '.join(x.split()[1:]) for x in attack_types]
    print(attack_types)
    plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')
    # plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=False, title='Normalized confusion matrix')

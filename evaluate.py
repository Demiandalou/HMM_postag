import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from argparse import ArgumentParser
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn


parser = ArgumentParser()

parser.add_argument("-p", "--predicted", dest = "pred_path",
    required = True, help = "path to your model's predicted labels file")

parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")

parser.add_argument("-c", "--confusion", dest = "show_confusion",
    action = "store_true", help = "show confusion matrix")

args = parser.parse_args()


pred = pd.read_csv(args.pred_path, index_col = "id")
dev  = pd.read_csv(args.dev_path,  index_col = "id")

pred.columns = ["predicted"]
dev.columns  = ["actual"]

data = dev.join(pred)


if args.show_confusion:
    
    data["count"] = 1
    counts = data.groupby(["actual", "predicted"]).count().reset_index()
    confusion = counts[counts.actual != counts.predicted].reset_index(drop = True)
        
    print("Confusion Matrix:")
    confusion.sort_values(by=['count'],ascending=False,inplace=True,
                     na_position='first')
    confusion.reset_index(drop=True,inplace=True)
    
    if confusion.empty: print("None!")
    else: print(confusion)

    df = confusion[:15]
    pred_list = list(df['predicted'])
    actual_list = list(df['actual'])
    for i in range(len(df)):
        if df['actual'][i] not in pred_list:
            df.drop(labels =i,inplace=True)
        if df['predicted'][i] not in actual_list:
            df.drop(labels =i,inplace=True)
    df.reset_index(drop=True,inplace=True)
    # print(df)

    # sample_actual, sample_predicted
    label_set = list(set(df['actual']))
    label_set.sort()
    mat = np.zeros((len(label_set), len(label_set)))
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            # print(label_set[i],label_set[j])
            tmp = confusion[(confusion['actual']==label_set[i]) & (confusion['predicted']==label_set[j])]
            tmp.reset_index(drop=True,inplace=True)
            if len(tmp)==0:
                mat[i,j] = 0
            else:
                mat[i,j] = tmp['count'][0]

    sns.heatmap(mat, square=True, annot=True,  cbar=False,fmt='g', # fmt='d',
                xticklabels=label_set,
                yticklabels=label_set)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.savefig('confusion_mat.png',dpi=500)
    plt.show()


else:

    print("Mean F1 Score:", f1_score(
        data.actual,
        data.predicted,
        average = "weighted"
    ))


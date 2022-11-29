import csv
from sklearn.metrics import confusion_matrix as cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS

from sklearn.linear_model import LogisticRegression as LRC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
model_dict = {'log': LRC, 'rfc': RFC, 'knn': KNC, 'svc': SVC}

import math

def split_and_scale(X, y):
    _X, X_, _y, y_ = tts(X, y)
    ss = SS()
    _Xs = ss.fit_transform(_X)
    Xs_ = ss.transform(X_)
    return _Xs, Xs_, _y, y_

# returns length N series (generator) of thresholds in increments of 1/N
def Hs(N):
    return (i/N for i in range(N+1))

# computes 'integral' of 'Y of X' using trapazoid rule for approximation
def auc(X, Y):
#   each 'dx' in 'dX' is the change in x value for each interval
    dX = [b - a for a, b in zip([0] + X, X + [0])][1:-1]
#   each 'foy' in 'foY' is the integrand - f(a) + f(b) /2 - for each interval
    foY = [(yoa + yob)/2 for yoa, yob in zip([0] + Y, Y + [0])][1:-1]
#   the return is the sum of the areas of each trapazoid, absolute value is taken as the direction of integration is not know
    return abs(sum(foy * dx for foy, dx in zip(foY, dX)))
    
# returns the list of 
def roc(X_, y_, model_inst, N, plot = False, area=False, save_path=None):
#   get thresholds
    H = Hs(N)
#   initialize lists
    F = []
    T = []
#   get probabilities of positives
    y_p = model_inst.predict_proba(X_)[:,1]
#   loop through all thresholds
    for t in H:
#       generate false positive and true positve rates: fpr, tpr
        tn, fp, fn, tp = cm(y_, list(map(lambda p: 1 if p >= t else 0, y_p))).ravel()
        fpr = fp/(tn+fp)
        tpr = tp/(tp+fn)
#       append lists
        F.append(fpr)
        T.append(tpr)
#   plot if necessary
    if plot == True:
        plt.plot(F, T)
        plt.xlabel('False Positve Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {str(model)[:-2]} Model')
        plt.show()
#   save to path name 'save_path' if provided
    if not save_path == None:
        with open(save_path, 'w') as f:
            csv.writer(f, lineterminator='\n').writerows(zip(F, T))
#   return area under curve if auc is selected
    if area:
        return auc(F, T)
#   otherwise return the fpr and tpr values
    else:
        return (F, T)

def acc_test(X, y, model='log', inst_num=None, **kwargs):
    y = y.to_numpy().ravel()
    _Xs, Xs_, _y, y_ = split_and_scale(X, y)
    M = model_dict[model]
    m = M(**kwargs)
    m.fit(_Xs, _y)
    return m.score(Xs_, y_)
    
def auc_test(X, y, model='log', inst_num=None, trials=False, **kwargs):
    y = y.to_numpy().ravel()
    _Xs, Xs_, _y, y_ = split_and_scale(X, y)
    M = model_dict[model]
    m = M(**kwargs)
    m.fit(_Xs, _y)
    if trials:
        return roc(Xs_, y_, m, 100, area=True)
    save_path = f'Outputs/{str(m).split("(")[0]}_ROC'
    if not inst_num == None:
        save_path += f'_{inst_num}'
    return roc(Xs_, y_, m, 100, save_path=save_path+'.csv', area=True)

def get_data_dict(save_path=None):
    import pandas as pd
    from splinter import Browser
    from webdriver_manager.chrome import ChromeDriverManager
    executable_path = {'executable_path': ChromeDriverManager().install()}
    with Browser('chrome', **executable_path, headless=False) as browser:
        browser.visit('http://www.kaggle.com/datasets/whenamancodes/credit-card-customers-prediction')
        data_dict = pd.read_html(browser.html)[0]
    pd.set_option('display.max_colwidth', None)
    if not save_path == None:
        data_dict.to_csv(save_path)
    return data_dict
import csv
from sklearn.metrics import confusion_matrix as cm
from matplotlib import pyplot as plt

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
def roc(X_, y_, model, N, plot = False, area=False, save_path=None):
#   get thresholds
    H = Hs(N)
#   initialize lists
    F = []
    T = []
#   get probabilities of positives
    y_p = model.predict_proba(X_)[:,1]
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
        plt.title(f'ROC for {str(type(model)).split(".")[-1][:-2]} Model')
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
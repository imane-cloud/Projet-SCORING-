I- Phase de prétraitement
Préparation de données :


import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns
dataset = pd.read_csv('ScoringTraining.csv')
dataset.head()

print(dataset.shape)


1-Proportion de défauts :

#La proportion de défaut est un ratio calculant le pourcentage de non-paiement des échéances d'un emprunt.
#Pour cela, on utilisera la 1ère colonne SeriousDlqin2years qui indique si la personne a des problèmes financières pendant
# les 3 mois derniers 
dataset["SeriousDlqin2yrs"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
labels = ['no Dlqin', 'Dlqin']
plt.legend(labels)

2-identification des outliers :
  
plt.subplot(121)
sns.boxplot(x=dataset["RevolvingUtilizationOfUnsecuredLines"])
plt.subplot(122)
sns.boxplot(x=dataset["DebtRatio"])
plt.show()

def find_all_outliers(v):
    Q1 = np.quantile(v, 0.25)  
    Q3 = np.quantile(v, 0.75)
    EIQ = Q3 - Q1
    LI = Q1 - (EIQ*1.5)
    LS = Q3 + (EIQ*1.5)    
    i = list(v.index[(v < LI) | (v > LS)])
    val = list(v[i])
    return i, val
  outliers = find_all_outliers(dataset['RevolvingUtilizationOfUnsecuredLines'])
#affichage d'une liste de valeurs des outliers de la variable RevolvingUtilizationOfUnsecuredLines
outliers[1] 
outliers = find_all_outliers(dataset["DebtRatio"])
#affichage d'une liste de valeurs des outliers de la variable DebtRatio
outliers[1]  


sns.boxplot(x=dataset["age"])

sns.boxplot(x=dataset["NumberOfTime30-59DaysPastDueNotWorse"])

sns.boxplot(x=dataset["MonthlyIncome"])

sns.boxplot(x=dataset["NumberOfOpenCreditLinesAndLoans"])

sns.boxplot(x=dataset["NumberOfTimes90DaysLate"])

sns.boxplot(x=dataset["NumberRealEstateLoansOrLines"])

sns.boxplot(x=dataset["NumberOfTime60-89DaysPastDueNotWorse"])

sns.boxplot(x=dataset["NumberOfDependents"])

dataset.boxplot(figsize=(40,15))

# voir s'il y a des valeurs manquantes 
dataset.isnull().sum()

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#correction de la colonne MonthlyIncome 
dataset['MonthlyIncome'].fillna(dataset['MonthlyIncome'].median(),inplace=True)
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#correction de la colonne NumberOfDependents
dataset['NumberOfDependents'].fillna(dataset['NumberOfDependents'].median(),inplace=True)
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

Equilibrage des données d’apprentissage :
5- choix des échantillons :
from collections import Counter
print(sorted(Counter(dataset['SeriousDlqin2yrs']).items()))

X=dataset
Y=dataset['SeriousDlqin2yrs']

from imblearn import under_sampling,under_sampling
from imblearn.under_sampling import RandomUnderSampler
ros=RandomUnderSampler(random_state=0)
data_resampled,y_resampled=ros.fit_resample(X,Y)
print(sorted(Counter(y_resampled).items()),y_resampled.shape)

y_resampled.value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
labels=['no dlqin','dlqin']
plt.legend()

dataset.boxplot(figsize=(40,15))

sns.boxplot(data=dataset,palette='pastel',orient='h')

Identification des meilleurs prédicteurs parmi les variables :
6-identification des meilleures variables dans le modèle de prédiction :
  
 # On utilisera la dataset équilibrée 
data_resampled

boites à moustache des deux classes pour chaque variable :
  
#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="RevolvingUtilizationOfUnsecuredLines",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="RevolvingUtilizationOfUnsecuredLines",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="age",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="age",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberOfTime30-59DaysPastDueNotWorse",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberOfTime30-59DaysPastDueNotWorse",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="DebtRatio",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="DebtRatio",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="MonthlyIncome",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="MonthlyIncome",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberOfOpenCreditLinesAndLoans",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberOfOpenCreditLinesAndLoans",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])


#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberOfTimes90DaysLate",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberOfTimes90DaysLate",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])


#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberRealEstateLoansOrLines",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberRealEstateLoansOrLines",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberOfTime60-89DaysPastDueNotWorse",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberOfTime60-89DaysPastDueNotWorse",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

#On sépare le dataset par groupe
grouped=data_resampled.groupby(dataset.SeriousDlqin2yrs)
data_0=grouped.get_group(0)
data_1=grouped.get_group(1)
ax=plt.axes()
#1ere boite à moustaches
boxplot=data_0.boxplot(column="NumberOfDependents",positions=[1],widths=0.6,showfliers=False)
#2eme boite à moustaches
boxplot=data_1.boxplot(column="NumberOfDependents",positions=[2],widths=0.6,showfliers=False)

ax.set_xticklabels(["0","1"])

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
clean_data,y_resampled=rus.fit_resample(X,Y)
print(sorted(Counter(y_resampled).items()),y_resampled.shape)

II- Modèle de prévision :
logistic Regression

logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)

#
y_pred=logreg.predict(x_test)

8-Application du LDA

import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

Y=clean_data.iloc[:,1]
X=clean_data.iloc[:,1:11]
lda=LinearDiscriminantAnalysis(n_components=1)
score=lda.fit(X,Y).transform(X)
y_pred=lda.predict(X)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3,random_state=1)
scores=cross_val_score(lda, X,Y,scoring="accuracy",cv=cv, n_jobs=-1)
print("Mean Accuracy: %.3f (%.3f)" % (np.mean(scores),np.std(scores)))
random_choices = [random.randint(0,len(score) -1) for i in range(0,300)]
score1=[score[i] for i in random_choices]
y_train1=[Y[i] for i in random_choices]
for k, i in zip(score1, y_train1) :
    if i==1:
        plt.scatter(k,0,c="b")
    else : 
       plt.scatter(k,0,c="r")
plt.xlabel("projection vector")
plt.show

8- Application du QDA

import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score

%matplotlib inline
y_train=clean_data.iloc[:, 1]
X_train=clean_data.iloc[:, 2:11]

y_test=clean_data.iloc[:, 1]
X_test=clean_data.iloc[:, 2:11]
qda = QuadraticDiscriminantAnalysis()
model2 = qda.fit(X_train, y_train)
print(model2.priors_)
print(model2.means_)
pred2=model2.predict(X_test)
print(np.unique(pred2, return_counts=True))
print(confusion_matrix(pred2, y_test))
print(classification_report(y_test, pred2, digits=3))

Autre model de prévison:
  
 #arbre de décision
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

y_train=clean_data.iloc[:,1]
x_train=clean_data.iloc[:,2:11]
clf = DecisionTreeRegressor(max_depth=2)
#Entrainement de l'abre de décision 
clf.fit(x_train, y_train.ravel())
#Affichage de l'abre de décision obtenu après entraînement
plot_tree(clf, filled=True)
plt.show()

Phase d’évaluation et règle de décision retenue
11-Goodness of fit et courbes de ROC :
  
 # la courbe de Roc de la regression logistic
y_train=clean_data.iloc[:,1]
x_train=clean_data.iloc[:,1:11]
y_test=data_resampled.iloc[:,1]
x_test=data_resampled.iloc[:,1:11]

from sklearn.linear_model import LogisticRegression
logistic_reg=LogisticRegression()
logistic_reg.fit(x_train, y_train.ravel())
y_prob=logistic_reg.predict_proba(x_test)[:,1]
y_pred=np.where(y_prob > 0.5 , 1 , 0)

false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc
def plot_roc(roc_auc):
    plt.figure(figsize=(7,7))
    plt.title('ROC LDA')
    plt.plot(false_positive_rate,true_positive_rate,color='red',label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')

    
    plot_roc(roc_auc)
    
    
    
 # Roc de Lda

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(x_train, y_train.ravel())
y_prob_lda=lda.predict_proba(x_test) [:,1]
y_pred_lda=np.where(y_prob_lda>0.5, 1,0)
lda_confusion_matrix = confusion_matrix (y_test,y_pred_lda)
lda_confusion_matrix
false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test,y_prob_lda)
roc_auc_lda=auc(false_positive_rate,true_positive_rate)
plot_roc(roc_auc_lda)


false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test,y_prob_lda)
roc_auc_lda=auc(false_positive_rate,true_positive_rate)
roc_auc_lda

def plot_roc(roc_auc):
    plt.figure(figsize=(7,7))
    plt.title('courbe ROC QDA')
    plt.plot(false_positive_rate,true_positive_rate,color='red',label='AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    
false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,pred2)
roc_auc_qda=auc(false_positive_rate,true_positive_rate)
roc_auc_qda

plot_roc(roc_auc_qda)

#matrice de confusion de linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
%matplotlib inline

log_confusion_matrix=confusion_matrix(y_test,y_pred)
log_confusion_matrix


false_positive_rate, true_positive_rate, thresholds=roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate,true_positive_rate)
roc_auc

#matrice de confusion LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(x_train, y_train.ravel())
y_prob_lda=lda.predict_proba(x_test) [:,1]
y_pred_lda=np.where(y_prob_lda>0.5, 1,0)
lda_confusion_matrix = confusion_matrix (y_test,y_pred_lda)
lda_confusion_matrix


#matrice de confusion QDA
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score

%matplotlib inline
y_train=clean_data.iloc[:, 1]
X_train=clean_data.iloc[:, 2:11]

y_test=clean_data.iloc[:, 1]
X_test=clean_data.iloc[:, 2:11]
qda = QuadraticDiscriminantAnalysis()
model2 = qda.fit(X_train, y_train)
pred2=model2.predict(X_test)
#print(np.unique(pred2, return_counts=True))
print(confusion_matrix(pred2, y_test))
#print(classification_report(y_test, pred2, digits=3))

         

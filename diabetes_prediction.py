import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sklearn.metrics as metrics


Diabetes= pd.read_csv('./diabetes_data_upload.csv') # cARGAR DATA SET
## print(Diabetes.head(10)) Muestra los principales 10 


Diabetes.isnull().sum()
Diabetes.isna().sum()
## Diabetes.info() MUESTRAEL TIPO Y SU CONTENIDO
Diabetes.describe()


#PROCESAMIENTO DE DATOS

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score,confusion_matrix,recall_score,roc_auc_score
from sklearn import model_selection

number = preprocessing.LabelEncoder()

dtacpy1 = Diabetes.copy()   # Duplicating the Dataset 
dtacpy1.head(5)

for i in dtacpy1:
    dtacpy1[i] = number.fit_transform(dtacpy1[i])

dtacpy1.head()

# Setting target variable 
X = dtacpy1.drop(['class'],axis=1) # Independent 
Y = dtacpy1['class'] # Dependent

X.head()
Y.head()


correlation = X.corrwith(Y)


correlation.plot.bar(title="Correlation with target variable class", grid=True, figsize=(15,5))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify= Y, random_state = 1000)

## checking for the distribution of traget variable in train test split




#normalizacion de datos

min_max = MinMaxScaler()
X_train[['Age']] = min_max.fit_transform(X_train[['Age']])
X_test[['Age']] = min_max.transform(X_test[['Age']])

X_train.head()
print("CHECK 4")
                       # CONSTRUCCION DEL MODELO
#regresion logistica
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split( X, Y, test_size = 0.3, random_state = 100)

logmodel = LogisticRegression()
logmodel.fit(X_train_LR, y_train_LR)

predictions_LR = logmodel.predict(X_test_LR)

final_model_predictions_LR = pd.DataFrame({'Actual':y_test_LR, 'predictions':predictions_LR})

print(confusion_matrix(y_test_LR, predictions_LR))
print(classification_report(y_test_LR, predictions_LR))
print("----------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------")
accuracy_LR=metrics.accuracy_score( final_model_predictions_LR.Actual, final_model_predictions_LR.predictions )*100
accuracy_LR='{:.2f}'.format(accuracy_LR)
print( 'Total Accuracy : ',accuracy_LR)
recall_LR=metrics.recall_score(final_model_predictions_LR.Actual, final_model_predictions_LR.predictions,average='micro' )
print('recall',recall_LR)
Precision_LR=metrics.precision_score(final_model_predictions_LR.Actual, final_model_predictions_LR.predictions,average='micro' )
print('Precision',Precision_LR)

print("CHECK 5")
#clasificador de arbol

from sklearn.tree import DecisionTreeClassifier 

decision_Tree_Classifier = DecisionTreeClassifier (random_state = 0) 

X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split( X, Y, test_size = 0.3, random_state = 100)

decision_Tree_Classifier.fit(X_train_DT, y_train_DT) 

# predicting a new value 
  
# test the output by changing values, like 3750 
y_pred_DT = decision_Tree_Classifier.predict(X_test_DT) 

final_model_predictions_DT = pd.DataFrame({'Actual':y_test_DT, 'predictions':y_pred_DT})

final_model_predictions_DT.head()

print(confusion_matrix(y_test_DT, y_pred_DT))
print(classification_report(y_test_DT, y_pred_DT))

accuracy_DT=( metrics.accuracy_score( final_model_predictions_DT.Actual, final_model_predictions_DT.predictions  ))*100
accuracy_DT='{:.2f}'.format(accuracy_DT)
print( 'Total Accuracy : ',accuracy_DT)
recall_DT=metrics.recall_score(final_model_predictions_DT.Actual, final_model_predictions_DT.predictions,average='micro' )
print('recall',recall_DT)
Precision_DT=metrics.precision_score(final_model_predictions_DT.Actual, final_model_predictions_DT.predictions,average='micro' )
print('Precision',Precision_DT)


print("CHECK 6")
##CLASIFICADOR DE VECTORES

X_train_SVC, X_test_SVC, y_train_SVC, y_test_SVC = train_test_split( X, Y, test_size = 0.3, random_state = 100)

from sklearn.svm import SVC # "Support Vector Classifier" 
clfsvm = SVC(kernel='linear') 

# fitting x samples and y classes 
clfsvm.fit(X_train_SVC.values,y_train_SVC) 



y_pred_SVC=clfsvm.predict(X_test_SVC)

final_model_predictions_SVC = pd.DataFrame({'Actual':y_test_SVC, 'predictions':y_pred_SVC})

# how did our model perform?
count_misclassified = (y_test_SVC != y_pred_SVC).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy_SVC = metrics.accuracy_score(y_test_SVC, y_pred_SVC)
print('Accuracy: {:.2f}'.format(accuracy_SVC))

print(confusion_matrix(y_test_SVC, y_pred_SVC))
print(classification_report(y_test_SVC, y_pred_SVC))
print("-----------------------------------------------------------------------------------------------------------------------")

print("-----------------------------------------------------------------------------------------------------------------------")
accuracy_SVC=metrics.accuracy_score( final_model_predictions_SVC.Actual, final_model_predictions_SVC.predictions  )*100
accuracy_SVC='{:.2f}'.format(accuracy_SVC)
print( 'Total Accuracy : ',accuracy_SVC)
recall_SVC=metrics.recall_score(final_model_predictions_SVC.Actual, final_model_predictions_SVC.predictions,average='micro' )
print('recall',recall_SVC)
Precision_SVC=metrics.precision_score(final_model_predictions_SVC.Actual, final_model_predictions_SVC.predictions,average='micro' )
print('Precision',Precision_SVC)

table=pd.DataFrame({"Accuracy":[accuracy_DT,accuracy_SVC,accuracy_LR],
                    "Recall":[recall_DT,recall_SVC,recall_LR],
                    "Precision ":[Precision_DT,Precision_SVC,Precision_LR]},
                   index=["Decision Tree Classifier","Support Vector Classifier","Logistic Regression"])



##RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf = RandomForestClassifier(n_estimators = 500, random_state = 42)

clf.fit(X_train_RF.values, y_train_RF);

import pickle


pickle.dump(clf,open('model.pkl','wb'))

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# use the model to make predictions with the test data
y_pred_RF = clf.predict(X_test_RF)
print(f"CONTROL : {y_pred_RF}")

# combining 2 numpy arrays into one pandas dataframe
final_model_predictions_RF = pd.DataFrame({'Actual':y_test_RF, 'predictions':y_pred_RF})

final_model_predictions_RF.head()

y_pred_prob = clf.predict_proba(X_test_RF)  # 2  columns for probability it is creating

y_pred_prob = clf.predict_proba(X_test_RF)[:,1]   # The first index refers to the probability that the data belong to class 0, and the second refers to the probability that the data belong to class 1

final_model_predictions_RF['Predicted_prob'] = y_pred_prob

final_model_predictions_RF['Predicted_prob'] = y_pred_prob

final_model_predictions_RF.head()

def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted )
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Default", "No Default"] , yticklabels = ["Default", "No Default"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()  # correct 0 is sensitivity and correct is specificity
    
draw_cm( final_model_predictions_RF.Actual, final_model_predictions_RF.predictions )   # correct 0 is sensitivity and correct is specificity

accuracy_RF=metrics.accuracy_score( final_model_predictions_RF.Actual, final_model_predictions_RF.predictions  )*100
accuracy_RF='{:.2f}'.format(accuracy_RF)
print( 'Total Accuracy : ',accuracy_RF )
recall_RF=metrics.recall_score(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions )
print('recall :',recall_RF)
precision_RF=metrics.precision_score(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions )
print('Precision :',precision_RF)

cm1 = metrics.confusion_matrix( final_model_predictions_RF.Actual, final_model_predictions_RF.predictions)

sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', round( sensitivity, 2) )

specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', round( specificity, 2 ) )



table=pd.DataFrame({"Accuracy":[accuracy_RF,accuracy_DT,accuracy_SVC,accuracy_LR],
                    "Recall":[recall_RF,recall_DT,recall_SVC,recall_LR],
                    "Precision ":[precision_RF,Precision_DT,Precision_SVC,Precision_LR]},
                   index=["Random Forest","Decision Tree Classifier","Support Vector Classifier","Logistic Regression"])
table


# ## ALGORITHMO DE VECINOS
# X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=2)
# classifier.fit(X_train_KNN, y_train_KNN)

# y_pred_knn = classifier.predict(X_test_KNN)

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test_KNN, y_pred_knn))
# print(classification_report(y_test_KNN, y_pred_knn))

# final_model_predictions_knn = pd.DataFrame({'Actual':y_test_KNN, 'predictions':y_pred_knn})

# draw_cm( final_model_predictions_knn.Actual, final_model_predictions_knn.predictions )   # correct 0 is sensitivity and correct is specificity

# accuracy_knn=metrics.accuracy_score( final_model_predictions_knn.Actual, final_model_predictions_knn.predictions)*100
# accuracy_knn='{:.2f}'.format(accuracy_knn)
# print( 'Total Accuracy : ',accuracy_knn)
# recall_knn=metrics.recall_score(final_model_predictions_knn.Actual, final_model_predictions_knn.predictions )
# print('recall',recall_knn)
# Precision_knn=metrics.precision_score(final_model_predictions_knn.Actual, final_model_predictions_knn.predictions )
# print('Precision',Precision_knn)

# cm2 = metrics.confusion_matrix( final_model_predictions_knn.Actual, final_model_predictions_knn.predictions)

# sensitivity = cm2[0,0]/(cm2[0,0]+cm2[0,1])
# print('Sensitivity : ', round( sensitivity, 2) )

# specificity = cm2[1,1]/(cm2[1,0]+cm2[1,1])
# print('Specificity : ', round( specificity, 2 ) )


# table=pd.DataFrame({"Accuracy":[accuracy_RF,accuracy_knn,accuracy_DT,accuracy_SVC,accuracy_LR],
#                     "Recall":[recall_RF,recall_knn,recall_DT,recall_SVC,recall_LR],
#                     "Precision ":[precision_RF,Precision_knn,Precision_DT,Precision_SVC,Precision_LR]},
#                    index=["Random Forest","KNN","Decision Tree Classifier","Support Vector Classifier","Logistic Regression"])


    
# import pickle
# import json

# pickle.dump(clf,open('model.pkl','wb'))

# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)


# def serialize_tree(tree):
#     tree_ = tree.tree_
#     return {
#         'nodes': tree_.node_count,
#         'children_left': tree_.children_left.tolist(),
#         'children_right': tree_.children_right.tolist(),
#         'feature': tree_.feature.tolist(),
#         'threshold': tree_.threshold.tolist(),
#         'value': tree_.value.tolist(),
#     }

# model_params = {
#     'n_estimators': model.n_estimators,
#     'estimators': [serialize_tree(estimator) for estimator in model.estimators_]
# }

# with open('model.json', 'w') as f:
#     json.dump(model_params, f)
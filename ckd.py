import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
#data-preprocessing

#loading data
df = pd.read_csv('ckd.csv')
df=df.drop(['id'],axis=1)

#converting alphabetic attributes into numerical value
le = preprocessing.LabelEncoder()
for column in df.columns:
        if df[column].dtype == object:
            df[column] = le.fit_transform(df[column].astype(str))

#replacing NaN values with mean
column_means = df.mean()
df = df.fillna(column_means)


#feature selection
X = df.iloc[:,:-1]
y = df['classification']
X_train, x_test, Y_train, y_test = train_test_split(X,y, random_state=0 ,test_size=0.30,shuffle=True)
svc_lin=SVC(kernel='linear')
svm_rfe_model=RFE(estimator=svc_lin)
svm_rfe_model_fit=svm_rfe_model.fit(X_train,Y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = X_train.columns)
signi_feat_rfe = feat_index[feat_index==1].index

#train-test-split
X = df[['bp', 'al', 'su', 'rbc', 'pc', 'bgr', 'sc', 'pot','hemo','htn','appet','pe']]
y = df['classification']
X_train, x_test, Y_train, y_test = train_test_split(X,y, random_state=0 ,test_size=0.30,shuffle=True)

#after feature selection using linear svm
svm_classifier = SVC(kernel='linear',random_state=0,)
svm_classifier.fit(X_train,Y_train)
# y_pred_svm_aft = svm_classifier.predict(x_test)

#after feature selection using decision tree algo
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
y_pred_dt_aft= model.predict(x_test)

#after feature selection using logistic regression algo
lr_basemodel =LogisticRegression()
lr_basemodel.fit(X_train,Y_train)
y_pred_lr_aft=lr_basemodel.predict(x_test)

pickle.dump(svm_classifier,open('ckd.pkl','wb'))
# Importing necessary libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Creating a pandad dataframe from CSV file
bank = pd.read_csv("bank-full.csv", sep=';')

# View data
print(bank.head())

print(bank.shape) #(45211, 17)

print(bank.isnull().any())

print(bank.dtypes)

print(bank.describe())


# Label encoding
# apply "le.fit_transform"
data = bank.copy(deep=True)
le = LabelEncoder()
def label_encoder(df):
    for columnName, columnData in df.iteritems():
        if df[columnName].dtype == object:
#             print(df[columnName])
#             df[columnName] = df[columnName].astype('category')
            df[columnName] = le.fit_transform(df[columnName])
        
# Driver code
label_encoder(data)
print(data.head())

print(data.dtypes)


# identifying independent and dependent variables
x = data.iloc[:, data.columns!='y']
y = data.iloc[:, data.columns=='y']
# print(x.head(), y.head())

# Model Building
logit = sm.Logit(endog=y,exog=x)
res = logit.fit()
print(res.summary())

y_pred = res.predict(x)

data["pred_prob"] = y_pred

data["y_predicted"] = np.zeros(data.shape[0])

# print(data.iloc[:,-3:])

data.loc[y_pred>=0.5,"y_predicted"] = 1
# print(data.iloc[:,-3:])

print("Classification Report :\n",classification_report(data["y_predicted"],data["y"]))

# confusion matrix 
confusion_matrix = pd.crosstab(data['y'],data["y_predicted"])
print("Confusion Matrix :\n",confusion_matrix)

confusion_matrix
accuracy = sum(np.diagonal(confusion_matrix))/data.shape[0]

print("Model Accuracy: ",accuracy)
# 88.95 Model accuracy

fpr, tpr, threshold = metrics.roc_curve(data['y'], y_pred)

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
plt.show()
 
roc_auc = metrics.auc(fpr, tpr) 
print("AUC: ",roc_auc)

# Dividing the train and test data sets
data.drop("y_predicted", axis=1, inplace=True)
data.drop("pred_prob", axis=1, inplace=True)
data.head()
x=data.iloc[:, data.columns!='y']
y=data.iloc[:, data.columns=='y']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=24)

model = sm.Logit(y_train,X_train)

res_train = model.fit()

#summary
print(res_train.summary())
train_pred = res_train.predict(X_train)

# filling all the cells with zeroes
X_train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
X_train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(y_train['y'],X_train.train_pred)

print(confusion_matrix)
accuracy_train = sum(np.diagonal(confusion_matrix))/X_train.shape[0] # 88.78
print("Training Accuracy: ",accuracy_train)


# Prediction on Test data set
test_pred = res_train.predict(X_test)

# Creating new column for storing predicted values

# filling all the cells with zeroes
X_test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
X_test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(y_test['y'],X_test.test_pred)

confusion_matrix
accuracy_test = sum(np.diagonal(confusion_matrix))/X_test.shape[0] # 89.30
print("Test Accuracy: ", accuracy_test)







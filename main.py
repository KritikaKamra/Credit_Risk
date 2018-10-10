import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC


train_df = pd.read_csv('Credit - Retail Banking Data - Train Data.csv')
test_df    = pd.read_csv('Credit - Retail Banking Data - Test Data.csv')
submission_df=pd.read_csv('Credit - Retail Banking Data - Submission Data.csv')

train_df['Securities_Account'] = train_df['Securities_Account'].replace('Yes', 1)
train_df['Securities_Account'] = train_df['Securities_Account'].replace('No', 0)
train_df['CD_Account'] = train_df['CD_Account'].replace('Yes', 1)
train_df['CD_Account'] = train_df['CD_Account'].replace('No', 0)
train_df['Online_Services'] = train_df['Online_Services'].replace('Yes', 1)
train_df['Online_Services'] = train_df['Online_Services'].replace('No', 0)
train_df['CreditCard'] = train_df['CreditCard'].replace('Yes', 1)
train_df['CreditCard'] = train_df['CreditCard'].replace('No', 0)
train_df['PersonalLoan (prediction)'] = train_df['PersonalLoan (prediction)'].replace('Yes', 1)
train_df['PersonalLoan (prediction)'] = train_df['PersonalLoan (prediction)'].replace('No', 0)

test_df['Securities_Account'] = test_df['Securities_Account'].replace('Yes', 1)
test_df['Securities_Account'] = test_df['Securities_Account'].replace('No', 0)
test_df['CD_Account'] = test_df['CD_Account'].replace('Yes', 1)
test_df['CD_Account'] = test_df['CD_Account'].replace('No', 0)
test_df['Online_Services'] = test_df['Online_Services'].replace('Yes', 1)
test_df['Online_Services'] = test_df['Online_Services'].replace('No', 0)
test_df['CreditCard'] = test_df['CreditCard'].replace('Yes', 1)
test_df['CreditCard'] = test_df['CreditCard'].replace('No', 0)
test_df['PersonalLoan'] = test_df['PersonalLoan'].replace('Yes', 1)
test_df['PersonalLoan'] = test_df['PersonalLoan'].replace('No', 0)

submission_df['Securities_Account'] = submission_df['Securities_Account'].replace('Yes', 1)
submission_df['Securities_Account'] = submission_df['Securities_Account'].replace('No', 0)
submission_df['CD_Account'] = submission_df['CD_Account'].replace('Yes', 1)
submission_df['CD_Account'] = submission_df['CD_Account'].replace('No', 0)
submission_df['Online_Services'] = submission_df['Online_Services'].replace('Yes', 1)
submission_df['Online_Services'] = submission_df['Online_Services'].replace('No', 0)
submission_df['CreditCard'] = submission_df['CreditCard'].replace('Yes', 1)
submission_df['CreditCard'] = submission_df['CreditCard'].replace('No', 0)

drop_elements = ['Age', 'Work_Exp', 'FamilySize', 'ZIP', 'Online_Services','CreditCard','SavingsID','Securities_Account']
test_df = test_df.drop(drop_elements, axis = 1)

s=[]
for i in (train_df['Edu_Level']):
    if i=='Higher Secondary':
        s.append(0)
    else:
        s.append(1)
k= pd.Series(s)
train_df['Graduated']=k.values

train_df = train_df.drop('Edu_Level', axis = 1)
test_df = test_df.drop('Edu_Level', axis = 1)
submission_df = submission_df.drop('Edu_Level', axis = 1)

s=[]
for i in (train_df['Annual_Income']):
    z=(i-80000)/(2050000-80000)
    s.append(z)
k= pd.Series(s)
train_df['Annual_Income']=k.values

s=[]
for i in (test_df['Annual_Income']):
    z=(i-80000)/(2240000-80000)
    s.append(z)
k= pd.Series(s)
test_df['Annual_Income']=k.values

s=[]
for i in (submission_df['Annual_Income']):
    z=(i-80000)/(2180000-80000)
    s.append(z)
k= pd.Series(s)
submission_df['Annual_Income']=k.values

s=[]
for i in (train_df['CC_Spend']):
    z=(i-0)/(240000)
    s.append(z)
k= pd.Series(s)
train_df['CC_Spend']=k.values

s=[]
for i in (test_df['CC_Spend']):
    z=(i-0)/(180000)
    s.append(z)
k= pd.Series(s)
test_df['CC_Spend']=k.values

s=[]
for i in (submission_df['CC_Spend']):
    z=(i-0)/(86000)
    s.append(z)
k= pd.Series(s)
submission_df['CC_Spend']=k.values

s=[]
for i in (train_df['OtherLoan_Monthly']):
    z=(i-0)/(186000)
    s.append(z)
k= pd.Series(s)
train_df['OtherLoan_Monthly']=k.values


s=[]
for i in (test_df['OtherLoan_Monthly']):
    z=(i-0)/(225000)
    s.append(z)
k= pd.Series(s)
test_df['OtherLoan_Monthly']=k.values

s=[]
for i in (submission_df['OtherLoan_Monthly']):
    z=(i-0)/(61200)
    s.append(z)
k= pd.Series(s)
submission_df['OtherLoan_Monthly']=k.values


X_train = train_df.drop("PersonalLoan (prediction)",axis=1)
Y_train = train_df["PersonalLoan (prediction)"]
X_test  = test_df.drop("PersonalLoan",axis=1).copy()
Y_test = test_df['PersonalLoan']

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = pd.Series(logreg.predict(X_test))
logreg.score(X_train,Y_train)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, Y_train)

drop_elements = ['SavingsID','PersonalLoan']

X_train = train_df.drop("PersonalLoan (prediction)",axis=1)
Y_train = train_df["PersonalLoan (prediction)"]
X_test  = submission_df.drop(drop_elements,axis=1).copy()
Y_test = submission_df['PersonalLoan']


random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


submission = pd.DataFrame({
        "SavingsID": submission_df["SavingsID"],
        "PersonalLoan": Y_pred
    })
submission.to_csv('Final.csv', index=False)
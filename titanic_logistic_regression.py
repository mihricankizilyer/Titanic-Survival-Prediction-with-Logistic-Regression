##############################################
# The Prediction of Titanic Passengers Survival
##############################################

"""
CONTENT:
 1. Reading Data
 2. Feature Engineering & Data Pre-Processing
        2.1 Feature Engineering
        2.2 Outliers
        2.3 Missing Values
        2.4 Label Encoding
        2.5 One-Hot Encoding
        2.6 Rare Encoding
        2.7 Standart Scaler
 3. Logistic Regression
        3.1 Model
        3.2 Prediction
        3.3 Success Evaluation
"""

import os
import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#############################################
# 1. READING DATA
#############################################

titanic_ = pd.read_csv('7.HAFTA/ödev/titanic.csv')
df = titanic_.copy()
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 2. FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

###########################
# 2.1 FEATURE ENGINEERING
###########################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

###########################
# 2.2 OUTLIERS
###########################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

# is there an outlier or not?
for col in num_cols:
    print(col, check_outlier(df, col))

# Thresholds values are retained and replaced with outlier values.
# Recovers from data loss in case of deletion.
for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)

###########################
# 2.3 MISSING VALUES
###########################

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,axis=0)

###########################
# 2.4 LABEL ENCODING
###########################
# Each data is assigned a unique integer in alphabetical order.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
            and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

###########################
# 2.5 RARE ENCODING
###########################
# It is an effort to consolidate the few classes.

df = rare_encoder(df, 0.01)

###########################
# 2.6 ONE-HOT ENCODING
###########################
# The main purpose is to meet the demands of the algorithms and to eliminate the measurement problems that may occur or to produce a higher quality data.

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

###########################
# 2.7 STANDART SCALER
###########################

scaler = StandardScaler().fit(df[["AGE"]])
df["AGE"] = scaler.transform(df[["AGE"]])
check_df(df)

#############################################
# 3. Logistic Regression
#############################################

###########################
# 3.1 Model
###########################

# LOG-REG:
# -> Dependent variable categorical
# -> Deals with class prediction (Alive or Not)

y = df["SURVIVED"] # the dependent variable
X = df.drop(["SURVIVED","PASSENGERID"], axis=1) # independent variable

# In order to better evaluate the prediction results
# The data set is divided into 2 as train-test.

# The model is installed on the train set.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state=1)
# Data is split from different places each time
# Divide by the same value each time with random_state
# So that it is tested with the same test data

# the model is installed
log_model = LogisticRegression().fit(X_train,y_train)

log_model.intercept_ # model constant -> beta value
# result -> array([-0.61632177])

log_model.coef_ # coefficients

###########################
# 3.2 Prediction
###########################
#Creating and saving forecasts

# The dependent variable is estimated over the independent variable.
y_pred = log_model.predict(X_train)

y_pred[0:10] # predicted values
y_test # real values

# class possaibilities
log_model.predict_proba(X_train[0:10])
# 0. index -> probability of being dead
# 1. index -> probability of survival

# [0.93550265, 0.06449735]
# [surviving, dead]

# If Threshold is 0.5
# 1. probability of occurrence 0
#2. probability of not happening 1

###########################
# 3.3 Success Evaluation
###########################

# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)
# result -> 0.8539325842696629

# Build the model with train, evaluate with test

# test
# y_pred for AUC Score
# Calculated for ROC curve
# Calculate the model used, probability, insert the test set.
y_prob = log_model.predict_proba(X_test)[:, 1]
# Probability of 1st class related to the dependent variable of test set
# And 1st class passed in trashold

# y_pred for other metrics
y_pred = log_model.predict(X_test)

# Confusion Matrix
def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=10))
    plt.show()

plot_confision_matrix(y_test, y_pred)

# precision = 53 / 65
# recall = 20 / 73

# ACCURACY
accuracy_score(y_test, y_pred)
# result -> 0.8212290502793296

# PRECISION
precision_score(y_test, y_pred)
# result -> 0.8153846153846154

# RECALL ( 53 / 53 + 20)
recall_score(y_test, y_pred) # estimate the value of what actually survived
# result ->  0.726027397260274

# F1
f1_score(y_test, y_pred)
# result ->  0.7681159420289856

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title("ROC CURVE")
plt.plot([0, 1],[0,1], 'r--')
plt.show()

# We can't evaluate the blue line alone so look at the AUC

# AUC
roc_auc_score(y_test, y_pred)
# result -> 0.8064099250452315

# Classification Reports
print(classification_report(y_test, y_pred))


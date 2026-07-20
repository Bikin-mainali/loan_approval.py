import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report

)


RANDOM_STATE = 42
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok = True)
sns.set_style("whitegrid")

#------------------------------ Loading the dataset ----------------------
#------------------------------
#------------------------------
df = pd.read_csv("previousApplicants.csv")
print("#---------------Loading the dataset -----------------------")
print(f"Raw shape:{df.shape}")  # total no of rows(13731) and cols(7)
print("Dataset loaded successfully!")

log = {} # it assembles the cleaning data for the report
log["raw-rows"] = len(df)


#------------------------------ 2.cleaning the dataset ----------------------

#-----------------------------2.1 Removing fully blank palceholder rows
sentinel_value = "999999999999999999999999"
fully_blank = df[df["Age"].isna() & df["Income"].isna() & df["Gender"].isna()]
log["fully_blank_rows_removed"] = len(fully_blank)
df = df.drop(fully_blank.index)


#-----------------------------2.2 Removing rows with sentinel value ----------------
sentinel_rows =  df[df["Outgoings"].astype(str) == sentinel_value]
log["sentinel_rows_removed"] = len(sentinel_rows)
df = df.drop(sentinel_rows.index)
df["Outgoings"] = pd.to_numeric(df["Outgoings"], errors="coerce")
print(f"Cleaned shape sentinel:{df.shape}")  


#-----------------------------2.3 Removing Duplicate rows -------------
df["Income_numeric_flag"] = pd.to_numeric(df["Income"], errors ="coerce").notna()
df= df.sort_values("Income_numeric_flag", ascending=False)
before = len(df)
df = df.drop_duplicates(subset= "ID", keep= "first")
log["duplicate_id_rows_removed"]=before - len(df)
df = df.drop(columns=["Income_numeric_flag"])
print(f"Cleaned shape duplicates:{df.shape}")  


#-----------------------------2.4 cleaning the Income Field------------
df["Income"]= pd.to_numeric(df["Income"].replace("nil",np.nan),errors = "coerce")
missing_income = df["Income"].isna().sum()
log["missing_income_after_cleaning"] = int(missing_income)
income_median = df["Income"].median()
df["Income"]= df["Income"].fillna(income_median)
print(f"Cleaned shape income:{df.shape}")


#-----------------------------2.5 cleaning the AGE-------------
missing_age = df["Age"].isna().sum()
log["missing_age_after_cleaning"] = int(missing_age)
df["Age"] = df["Age"].fillna(df["Age"].median())
print(f"Cleaned shape age:{df.shape}")  


#-----------------------------2.6 for the gender
log["gender_value_counts"]= df["Gender"].value_counts(dropna=False).to_dict()


#-------------------------2.7 Final check
log["rows_after_cleaning"] = len(df)
log["duplicate_full_rows_remaining"] = int(df.duplicated().sum())
assert df["Age"].between(18,100).all(),"invalid age"
assert (df["Income"]>=0).all(),"Negative income found"
assert (df["Outgoings"]>=0).all(), "Negative outgoings found"

print(f"Rows after cleaning:{len(df)}")


#-----------------------------3. FEATURE SELECTION----------
df["Outgoings_to_Income"] = df["Outgoings"]/df["Income"].replace(0,np.nan)
df["Outgoings_to_Income"] = df["Outgoings_to_Income"].fillna(df["Outgoings_to_Income"].median())

feature_cols=["Age","Income","Outgoings","Outgoings_to_Income"]
x= df[feature_cols].copy()
y = df["Approved"].astype(int)
log["feature_columns"]  = feature_cols
log["class_balance"] = y.value_counts(normalize=True).to_dict()


#---------------------------4.TRAIN / TEST SPLIT AND SCALING-----------------
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#---------------------------5. MODEL TRAINING -----------------
random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
random_forest.fit(x_train, y_train)
logistic_regression = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE,max_iter=1000)
logistic_regression.fit(x_train, y_train)


#---------------------------6. MODEL EVALUATION -----------------

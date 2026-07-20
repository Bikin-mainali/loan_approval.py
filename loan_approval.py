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
def evaluate(model, x_test, y_test, model_name):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    print(f"Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print(classification_report(y_test, predictions, target_names=["Rejected", "Approved"]))
    return metrics, predictions, probabilities

rf_metrics, rf_predictions, rf_probabilities = evaluate(random_forest, x_test, y_test, "Random Forest")
lr_metrics, lr_predictions, lr_probabilities = evaluate(logistic_regression, x_test, y_test, "Logistic Regression")

log["rf_metrics"] = rf_metrics
log["lr_metrics"] = lr_metrics


#---------------------------7. FIGURES FOR THE REPORT -----------------

#7.1 class balance 
plt.figure(figsize = (5,4))
y.value_counts().rename({0:"Rejected", 1:"Approved"}).plot(kind="bar", color=["#2ce679","#ee4836"])
plt.title("class balance of Approved Outcomes")
plt.ylabel("Number of Applications")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
plt.close()

#7.2 confusion matrices
fig, axes = plt.subplots(1,2, figsize=(10,4))
for ax, predictions, model_name in zip(axes, [rf_predictions, lr_predictions],["Random Forest", "Logistic Regression"]):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot = True, fmt="d", cmap="Blues", ax=ax, xticklabels= ["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    ax.set_title(model_name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()

#7.3 ROC Curves
plt.figure(figsize=(5.5, 5))
for probabilities, model_name in [(rf_probabilities, "Random Forest"), (lr_probabilities, "Logistic Regression")]:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label = f"{model_name} (AUC = {auc:.3f})")
plt.plot([0,1],[0,1], "k--", label = "Random classifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve Comparision")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()

#7.4 Random Forest feature importance
importances = pd.Series(random_forest.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(6,4))
importances.plot(kind="barh", color="#2980b9")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rf_feature_importance.png", dpi=150)
plt.close()
log["rf_feature_importance"] = importances.to_dict()

#7.5 Metric Comparison bar chart 
metrics_df = pd.DataFrame({"Random Forest": rf_metrics, "Logistic Regression": lr_metrics}).drop("roc_auc")
plt.figure(figsize=(6,4))
metrics_df.plot(kind="bar", ax=plt.gca(), color=["#2980b9","#27ae60"])
plt.title("Model Metrics Comparison on Test set")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_metrics_comparison.png", dpi=150)
plt.close()

#--------------------------------RESULTS LOG-------------------
with open(f"{OUT_DIR}/ run_log.json","w") as f:
    json.dump(log, f, indent= 2, default=str)

print("\nAll figures and run_log.json saved to outputs/")

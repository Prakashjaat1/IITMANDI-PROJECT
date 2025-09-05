# Cybersecurity Threat Detection using ML (NSL-KDD Dataset)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# 1. Load Dataset
# ======================
train = pd.read_csv("KDDTrain+_20Percent.csv")
test = pd.read_csv("KDDTest+.csv")

# Assign column names (from NSL-KDD documentation)
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

train.columns = col_names
test.columns = col_names

# Drop difficulty column
train = train.drop(["difficulty"], axis=1)
test = test.drop(["difficulty"], axis=1)

# ======================
# 2. Preprocessing
# ======================
# Encode categorical features
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Simplify labels (Normal vs Attack)
train["label"] = train["label"].apply(lambda x: "normal" if x=="normal" else "attack")
test["label"] = test["label"].apply(lambda x: "normal" if x=="normal" else "attack")

# Split features and labels
X_train, y_train = train.drop("label", axis=1), train["label"]
X_test, y_test = test.drop("label", axis=1), test["label"]

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 3. Train Model
# ======================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======================
# 4. Evaluation
# ======================
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


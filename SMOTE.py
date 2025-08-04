#%%
import pandas as pd
import numpy as np
import warnings
import pandas as pd 
import imblearn
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

#%%
import pandas as pd
import numpy as np
import warnings
import pandas as pd 

import matplotlib.pyplot as plt 
import numpy as np
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



df= pd.read_csv('C:/Users/PC/Desktop/bil 476/Proje/insurance_claims.csv')

# %%
df.drop(columns=['_c39'],inplace=True)
# %%
y=df['fraud_reported']
X = df.drop(columns=["fraud_reported"])
# %%

mode_value = X['authorities_contacted'].mode()[0]

X['authorities_contacted'].fillna(mode_value, inplace=True)
# %%

numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


datetime_columns = X.select_dtypes(include=['datetime64[ns]', 'datetime']).columns.tolist()


object_columns = X.select_dtypes(include=['object']).columns.tolist()


print("Numerik sütunlar:", numerical_columns)
print("Zaman sütunları:", datetime_columns)
print("Object/String sütunlar:", object_columns)

# %%

possible_date_columns = [col for col in X.select_dtypes(include='object').columns if 'date' in col.lower()]

for col in possible_date_columns:
    X[col] = pd.to_datetime(X[col], errors='coerce')  



# %%
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


datetime_columns = X.select_dtypes(include=['datetime64[ns]', 'datetime']).columns.tolist()


object_columns = X.select_dtypes(include=['object']).columns.tolist()
# %%

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].replace("?", np.nan)

nan_columns = X.columns[X.isnull().any()].tolist()

for column in X.columns:
    if X[column].isnull().any():
        mode_value = X[column].mode()[0]
        X[column].fillna(mode_value, inplace=True)
# %%
for col in datetime_columns:
    X[f"{col}_year"] = X[col].dt.year
    X[f"{col}_month"] = X[col].dt.month
    X[f"{col}_day"] = X[col].dt.day
    X[f"{col}_weekday"] = X[col].dt.weekday
    X[f"{col}_is_weekend"] = X[col].dt.weekday >= 5
    X[f"{col}_dayofyear"] = X[col].dt.dayofyear
    X[f"{col}_week"] = X[col].dt.isocalendar().week
    


# %%
columns_to_drop = [
    'policy_number',
    'insured_zip',
    'incident_location',
    'auto_model',
    'insured_hobbies',
    'policy_bind_date','incident_date'
]

X.drop(columns=columns_to_drop, axis=1, inplace=True)

# %%

X[['policy_csl_min', 'policy_csl_max']] = X['policy_csl'].str.split('/', expand=True).astype(int)

X.drop('policy_csl', axis=1, inplace=True)

# %%
label_encode_cols = [
    'insured_occupation',
    'incident_type',
    'collision_type',
    'authorities_contacted',
    'incident_state',
    'incident_city',
    'auto_make',
    'insured_relationship'
]
from sklearn.preprocessing import LabelEncoder

for col in label_encode_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

one_hot_encode_cols = [
    'policy_state',
    'insured_sex',
    'property_damage',
    'police_report_available'
]


X = pd.get_dummies(X, columns=one_hot_encode_cols, drop_first=True)
# %%
insured_education_order = [
    'High School',
    'Associate',
    'College',
    'Masters',
    'JD',
    'MD',
    'PhD'
]

incident_severity_order = [
    'Trivial Damage',  
    'Minor Damage',
    'Major Damage',
    'Total Loss'       
]
from sklearn.preprocessing import OrdinalEncoder


ordinal_encode_cols = {
    'insured_education_level': [insured_education_order],
    'incident_severity': [incident_severity_order]
}

for col, order in ordinal_encode_cols.items():
    oe = OrdinalEncoder(categories=order)
    X[col] = oe.fit_transform(X[[col]])

# %%
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)
#%%


y = y.map({'Y': 1, 'N': 0})

# %%
selector = VarianceThreshold(threshold=0)
selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]
# %%
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

while True:
    mut = pd.Series(mutual_info_classif(X, y), index=X.columns)
    sifir = mut[mut == 0].index.tolist()
    if not sifir:  
        break
    X = X.drop(columns=sifir)
# %%
def corr_mutual(X, mutual_info, threshold=0.9):

    corr_matrix = X.corr()
    features_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                if mutual_info.at[col1] < mutual_info.at[col2]:
                    features_to_drop.add(col1)
                else:
                    features_to_drop.add(col2)
 
    return features_to_drop

corr_features = corr_mutual(X, mut)


X.drop(columns=corr_features,inplace=True)
# %%
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, stratify=y,random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
space_xgboost = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),  
    'n_estimators': hp.quniform('n_estimators', 50, 550, 10),  
    'max_depth': hp.quniform('max_depth', 1, 15, 1),  
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),  
    'subsample': hp.uniform('subsample', 0.2, 1),
    'gamma': hp.uniform('gamma', 0, 5),  
    'max_delta_step': hp.uniform('max_delta_step', 0, 10),  
    'alpha': hp.uniform('alpha', 0, 5), 
    'lambda': hp.uniform('lambda', 0, 5),  
}

def objective_xgboost(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='gpu_hist',
        **params
    )

    clf.fit(X_train_scaled, y_train, verbose=False)
    pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, pred)

    return {'loss': -accuracy, 'status': STATUS_OK}

trials_xgboost = Trials()
best_hyperparams_xgboost = fmin(
    fn=objective_xgboost,
    space=space_xgboost,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials_xgboost
)

final_xgboost_params = {
    'learning_rate': best_hyperparams_xgboost['learning_rate'],
    'n_estimators': int(best_hyperparams_xgboost['n_estimators']),
    'max_depth': int(best_hyperparams_xgboost['max_depth']),
    'min_child_weight': best_hyperparams_xgboost['min_child_weight'],
    'subsample': best_hyperparams_xgboost['subsample'],
    'gamma': best_hyperparams_xgboost['gamma'],
    'max_delta_step': best_hyperparams_xgboost['max_delta_step'],
    'alpha': best_hyperparams_xgboost['alpha'],  
    'lambda': best_hyperparams_xgboost['lambda'],  
}

best_xgboost_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',
    **final_xgboost_params
)

best_xgboost_model.fit(X_train_scaled, y_train, verbose=False)
y_pred_xgboost = best_xgboost_model.predict(X_test_scaled)
xgboost_test_accuracy = accuracy_score(y_test, y_pred_xgboost)

print("Best XGBoost Hyperparameters:\n", final_xgboost_params)
print(f"XGBoost Test Set Accuracy: {xgboost_test_accuracy:.4f}")

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_xgboost_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("CV Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())


# %%
model_spaces = {
    'random_forest': {
        'n_estimators': hp.quniform('rf_n_estimators', 100, 500, 10),
        'max_depth': hp.quniform('rf_max_depth', 3, 30, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', None]),
        'bootstrap': hp.choice('rf_bootstrap', [True, False]),
        'model_type': 'random_forest'
    },
    'logistic_regression': {
        'C': hp.loguniform('lr_C', np.log(0.001), np.log(10)),
        'solver': hp.choice('lr_solver', ['lbfgs', 'saga']),
        'model_type': 'logistic_regression'
    },
    'svm': {
        'C': hp.loguniform('svm_C', np.log(0.001), np.log(10)),
        'gamma': hp.loguniform('svm_gamma', np.log(0.0001), np.log(1)),
        'kernel': hp.choice('svm_kernel', ['rbf', 'poly', 'sigmoid']),
        'model_type': 'svm'
    },
    'knn': {
        'n_neighbors': hp.quniform('knn_n_neighbors', 3, 30, 1),
        'weights': hp.choice('knn_weights', ['uniform', 'distance']),
        'algorithm': hp.choice('knn_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': hp.quniform('knn_leaf_size', 10, 60, 5),
        'model_type': 'knn'
    }
}
def objective(params):
    model_type = params.pop('model_type')

    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            random_state=42
        )

    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            C=params['C'],
            solver=params['solver'],
            max_iter=500
        )

    elif model_type == 'svm':
        model = SVC(
            C=params['C'],
            gamma=params['gamma'],
            kernel=params['kernel']
        )

    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=int(params['n_neighbors']),
            weights=params['weights'],
            algorithm=params['algorithm'],
            leaf_size=int(params['leaf_size'])
        )

    else:
        return {'loss': 1.0, 'status': STATUS_OK}

    acc = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    return {'loss': -acc, 'status': STATUS_OK}

#%%
results = {}
for model_name, space in model_spaces.items():
    print(f"Optimizing {model_name}...")
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    results[model_name] = best
# %%


# %%
from sklearn.utils import shuffle
import seaborn as sns
from hyperopt import space_eval

model_names = []
mean_accuracies = []

for model_name, best_params in results.items():
    space = model_spaces[model_name].copy()
    model_type = space.pop("model_type")
    
    full_params = space_eval(space, best_params)  # HP parametrelerini gerçek değerlere çevir
    full_params['model_type'] = model_type

    result = objective(full_params)
    model_names.append(model_name)
    mean_accuracies.append(-result['loss'])  # çünkü objective'de loss = -accuracy idi

model_names.append("xgboost")
mean_accuracies.append(xgboost_test_accuracy) 
# Bar chart
plt.figure(figsize=(10,6))
sns.barplot(x=model_names, y=mean_accuracies)
plt.title("Model Performance Comparison (CV/Test Accuracy)")
plt.ylabel("Mean Accuracy")
plt.xlabel("Model")
plt.ylim(0.7, 1.0)
plt.grid(True, axis='y')
plt.show()
# %%
# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay
import seaborn as sns

# Classification Report ve ROC AUC
print("Classification Report:\n", classification_report(y_test, y_pred_xgboost))
print("ROC AUC Score:", roc_auc_score(y_test, best_xgboost_model.predict_proba(X_test_scaled)[:, 1]))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_xgboost)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_xgboost_model.predict_proba(X_test_scaled)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, best_xgboost_model.predict_proba(X_test_scaled)[:, 1]):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid(True)
plt.show()
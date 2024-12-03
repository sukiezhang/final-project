# src/modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score
from sklearn.base import clone

def scale_features(df):
    """Scale only the numeric features using StandardScaler."""
    
    # Select only numeric columns (int and float types)
    df_numeric = df.select_dtypes(include=['number'])
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale the numeric features
    df_scaled = scaler.fit_transform(df_numeric)
    
    # Create a new DataFrame for the scaled features with the original column names
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
    
    return df_scaled, scaler

def train_extra_trees(X_train, y_train):
    """Train an ExtraTreesClassifier."""
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_eval, y_eval):
    """Evaluate the model and return accuracy and confusion matrix."""
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_eval, y_pred)
    cm = confusion_matrix(y_eval, y_pred)
    return acc, cm, y_pred

def compute_roc_auc(y_eval, y_pred):
    """Compute ROC curve and AUC score."""
    fpr, tpr, thresholds = roc_curve(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_pred)
    return fpr, tpr, roc_auc

def perform_kfold_cv(model, X, y, n_splits=10, random_state=42, scoring='accuracy'):
    """Perform K-Fold cross-validation and return scores."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return cv_scores


def get_feature_importances(model, X, y, k_folds=10, random_state=42):
    """Compute average feature importances using K-Fold cross-validation."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    feature_importances = np.zeros(X.shape[1])
    
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        cloned_model = clone(model)
        cloned_model.fit(X_train_fold, y_train_fold)
        feature_importances += cloned_model.feature_importances_
    
    feature_importances /= k_folds
    return feature_importances


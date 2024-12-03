# main.py

import os
import pandas as pd
import argparse  # Import other modules as before
from sklearn.model_selection import train_test_split

from src.data_loader import list_files, load_data
from src.data_processing import (
    split_blood_pressure,
    handle_missing_values,
    encode_features,
    preprocess_test_data
)
from src.data_exploration import (
    heart_attack_risk_distribution,
    age_distribution,
    sex_distribution,
    correlation_matrix
)
from src.visualization import (
    plot_socio_demographics,
    plot_clinical_characteristics,
    plot_lifestyle_behaviors,
    plot_comorbidities,
    plot_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importances
)
from src.modeling import (
    scale_features,
    train_extra_trees,
    train_random_forest,
    evaluate_model,
    compute_roc_auc,
    perform_kfold_cv,
    get_feature_importances
)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Automate data processing and visualization.')
    parser.add_argument('--figures', action='store_true', help='Generate figures only.')
    parser.add_argument('--tables', action='store_true', help='Generate tables only.')
    args = parser.parse_args()

    # Define paths
    data_dir = 'data/original_data'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    submission_path = os.path.join(data_dir, 'sample_submission.csv')

    output_figures = 'output/figures'
    output_tables = 'output/tables'

    os.makedirs(output_figures, exist_ok=True)
    os.makedirs(output_tables, exist_ok=True)

    if not args.tables:
        # List all files
        list_files('/Users/shuqizhang/Documents/zsq/UNC/BIOS 611/final_project')

    # Load data
    train, test, sample_submission = load_data(train_path, test_path, submission_path)

    # Check and clean column names
    print("Original column names in training data:", train.columns)
    train.columns = train.columns.str.replace(' ', '_')  # Replace spaces with underscores
    print("Updated column names in training data:", train.columns)

    if args.tables or not (args.figures or args.tables):
        # Save head of train and test to tables
        train.head().to_csv(f"{output_tables}/train_head.csv", index=False)
        test.head().to_csv(f"{output_tables}/test_head.csv", index=False)

        # Split Blood Pressure
        train = split_blood_pressure(train)
        train.head().to_csv(f"{output_tables}/train_blood_pressure_split.csv", index=False)

        # Missing values
        missing_values = handle_missing_values(train)
        missing_values.to_csv(f"{output_tables}/missing_values.csv", header=['Missing Values'])

        # Overall heart attack risk distribution
        heart_risk_dist = heart_attack_risk_distribution(train)
        heart_risk_dist.to_csv(f"{output_tables}/heart_attack_risk_distribution.csv")

        # Age distribution
        age_dist = age_distribution(train)
        age_dist.to_frame().to_csv(f"{output_tables}/age_distribution.csv")

        # Sex distribution
        sex_dist = sex_distribution(train)
        sex_dist.to_csv(f"{output_tables}/sex_distribution.csv")

    if args.figures or not (args.figures or args.tables):
        # Data visualization
        palette = ['#5aa2de', '#ff696e']  # Pink and Blue colors
        plot_socio_demographics(train, palette, output_figures)
        plot_clinical_characteristics(train, palette, output_figures)
        plot_lifestyle_behaviors(train, palette, output_figures)
        plot_comorbidities(train, palette, output_figures)

        # Correlation Matrix
        X = encode_features(train)
        corr = correlation_matrix(X)
        corr.to_csv(f"{output_tables}/correlation_matrix.csv")
        plot_heatmap(corr, output_figures)

    if not (args.figures or args.tables):
        # Prepare data for modeling
        # train_scaled, scaler = scale_features(train)
        # X_train, X_eval, y_train, y_eval = split_data(train, 'Heart_Attack_Risk')
        # train = train.drop(columns=['Patient ID','Blood Pressure','Country'], axis = 1) # these three variable won't be used in models.
        
        # X= train.drop(columns=["Heart_Attack_Risk", "Patient_ID", "Blood_Pressure","Country"],axis=1)
        X = encode_features(train)
        X, scaler = scale_features(X)
        y= train["Heart_Attack_Risk"]
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2,random_state=2024)


        # Train models
        Ext = train_extra_trees(X_train, y_train)
        RF = train_random_forest(X_train, y_train)

        # Evaluate ExtraTrees
        Ext_acc, Ext_cm, y_pred_Ext = evaluate_model(Ext, X_eval, y_eval)
        plot_confusion_matrix(Ext_cm, "ExtraTrees Confusion Matrix", output_figures)

        fpr_Ext, tpr_Ext, roc_auc_Ext = compute_roc_auc(y_eval, y_pred_Ext)
        plot_roc_curve(fpr_Ext, tpr_Ext, roc_auc_Ext, "Receiver Operating Characteristic Curve for ExtraTrees", output_figures)

        # Evaluate RandomForest
        RF_acc, RF_cm, y_pred_RF = evaluate_model(RF, X_eval, y_eval)
        plot_confusion_matrix(RF_cm, "RandomForest Confusion Matrix", output_figures)

        fpr_RF, tpr_RF, roc_auc_RF = compute_roc_auc(y_eval, y_pred_RF)
        plot_roc_curve(fpr_RF, tpr_RF, roc_auc_RF, "Receiver Operating Characteristic Curve for RandomForest", output_figures)

        # K-Fold Cross-Validation
        k_folds = 10
        cv_scores_Ext = perform_kfold_cv(Ext, X, y, n_splits=k_folds, scoring='accuracy')
        cv_f1_Ext = perform_kfold_cv(Ext, X, y, n_splits=k_folds, scoring='f1')

        cv_scores_RF = perform_kfold_cv(RF, X, y, n_splits=k_folds, scoring='accuracy')
        cv_f1_RF = perform_kfold_cv(RF, X, y, n_splits=k_folds, scoring='f1')

        # Save CV results
        pd.DataFrame({
            'ExtraTrees_Accuracy': cv_scores_Ext,
            'ExtraTrees_F1': cv_f1_Ext,
            'RandomForest_Accuracy': cv_scores_RF,
            'RandomForest_F1': cv_f1_RF
        }).to_csv(f"{output_tables}/cross_validation_results.csv", index=False)

        # Feature Importances for ExtraTrees
        feature_importances_Ext = get_feature_importances(Ext, X, y, k_folds=k_folds)
        feature_names = list(X_train.columns)
        plot_feature_importances(feature_importances_Ext, feature_names, "Average Feature Importances (ExtraTrees - 10-fold CV)", output_figures)

        # Feature Importances for RandomForest
        feature_importances_RF = get_feature_importances(RF, X, y, k_folds=k_folds)
        plot_feature_importances(feature_importances_RF, feature_names, "Average Feature Importances (RandomForest - 10-fold CV)", output_figures)

        print("All tasks completed successfully!")


if __name__ == '__main__':
    main()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_socio_demographics(train, palette, output_path):
    """Create socio-demographics plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    sns.histplot(train, x='Age', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 0], palette=palette)
    sns.countplot(data=train, x='Sex', hue='Heart_Attack_Risk', ax=axes[0, 1], palette=palette)
    sns.histplot(train, x='Income', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 2], palette=palette)
    sns.countplot(train, x='Country', hue='Heart_Attack_Risk', ax=axes[1, 0], palette=palette)
    sns.countplot(train, x='Continent', hue='Heart_Attack_Risk', ax=axes[1, 1], palette=palette)
    sns.countplot(data=train, x='Hemisphere', hue='Heart_Attack_Risk', ax=axes[1, 2], palette=palette)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/socio_demographics.png")
    plt.close()

def plot_clinical_characteristics(train, palette, output_path):
    """Create clinical characteristics plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    sns.histplot(train, x='BMI', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 0], palette=palette)
    sns.histplot(train, x='Cholesterol', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 1], palette=palette)
    sns.histplot(train, x='Blood_Pressure', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 2], palette=palette)
    sns.histplot(train, x='Heart_Rate', hue='Heart_Attack_Risk', kde=True, ax=axes[1, 0], palette=palette)
    sns.histplot(train, x='Triglycerides', hue='Heart_Attack_Risk', kde=True, ax=axes[1, 1], palette=palette)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/clinical_characteristics.png")
    plt.close()

def plot_lifestyle_behaviors(train, palette, output_path):
    """Create lifestyle and behaviors plots."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    sns.countplot(data=train, x='Smoking', hue='Heart_Attack_Risk', ax=axes[0, 0], palette=palette)
    sns.countplot(data=train, x='Alcohol_Consumption', hue='Heart_Attack_Risk', ax=axes[0, 1], palette=palette)
    sns.histplot(train, x='Exercise_Hours_Per_Week', hue='Heart_Attack_Risk', kde=True, ax=axes[0, 2], palette=palette)
    sns.countplot(data=train, x='Diet', hue='Heart_Attack_Risk', ax=axes[1, 0], palette=palette)
    sns.histplot(train, x='Sedentary_Hours_Per_Day', hue='Heart_Attack_Risk', kde=True, ax=axes[1, 1], palette=palette)
    sns.histplot(train, x='Physical_Activity_Days_Per_Week', hue='Heart_Attack_Risk', kde=True, ax=axes[1, 2], palette=palette)
    sns.histplot(train, x='Sleep_Hours_Per_Day', hue='Heart_Attack_Risk', kde=True, ax=axes[2, 0], palette=palette)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/lifestyle_behaviors.png")
    plt.close()

def plot_comorbidities(train, palette, output_path):
    """Create comorbidities plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    sns.countplot(data=train, x='Obesity', hue='Heart_Attack_Risk', ax=axes[0, 0], palette=palette)
    sns.countplot(data=train, x='Previous_Heart_Problems', hue='Heart_Attack_Risk', ax=axes[0, 1], palette=palette)
    sns.countplot(data=train, x='Medication_Use', hue='Heart_Attack_Risk', ax=axes[0, 2], palette=palette)
    sns.countplot(data=train, x='Stress_Level', hue='Heart_Attack_Risk', ax=axes[1, 0], palette=palette)
    sns.countplot(data=train, x='Diabetes', hue='Heart_Attack_Risk', ax=axes[1, 1], palette=palette)
    sns.countplot(train, x='Family_History', hue='Heart_Attack_Risk', ax=axes[1, 2], palette=palette)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/comorbidities.png")
    plt.close()

def plot_heatmap(df, output_path):
    """Plot and save the correlation heatmap."""
    
    # Ensure only numeric columns are considered for the correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])
    
    # Compute the correlation matrix
    corr = numeric_df.corr(method='pearson')
    
    # Check if the correlation matrix is empty or contains invalid data
    if corr.empty or corr.isnull().values.any():
        raise ValueError("Invalid correlation matrix: It is empty or contains NaN values.")
    
    # Create the heatmap figure
    plt.figure(figsize=(20, 20))
    
    # Plot the heatmap with annotations and format
    sns.heatmap(corr, cmap='YlGnBu', annot=True, fmt=".2f", cbar=True, square=True)
    
    # Rotate axis labels for better readability
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns, rotation=0)
    
    # Save the heatmap to the specified path
    plt.savefig(f"{output_path}/correlation_heatmap.png")
    plt.close()

def plot_confusion_matrix(cm, title, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, cmap='YlGnBu', annot=True, fmt='.0f')
    plt.xlabel("Predicted Outcome")
    plt.ylabel("True Outcome")
    plt.title(title)
    plt.savefig(f"{output_path}/{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, title, output_path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"{output_path}/{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_feature_importances(feature_importances, feature_names, title, output_path):
    """Plot and save feature importances."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances, tick_label=feature_names)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title.replace(' ', '_').lower()}.png")
    plt.close()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def analyze_target_variable(data: pd.DataFrame, target_column: str) -> str:
    """
    Analyze target variable to determine ML task type
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        
    Returns:
        String indicating the ML task type
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    target_data = data[target_column].dropna()
    
    if len(target_data) == 0:
        return "Unknown"
    
    # Check if target is numerica
    if pd.api.types.is_numeric_dtype(target_data):
        unique_values = target_data.nunique()
        total_values = len(target_data)
        
        # If very few unique values relative to total, likely classification
        if unique_values <= 10 or (unique_values / total_values) < 0.05:
            return "Classification"
        else:
            return "Regression"
    
    # If target is categorical/object
    else:
        unique_values = target_data.nunique()
        
        # Binary classification
        if unique_values == 2:
            return "Binary Classification"
        # Multi-class classification
        elif unique_values <= 20:
            return "Classification"
        else:
            # Too many categories, might be a text classification problem
            return "Text Classification"

def evaluate_data_readiness(data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
    """
    Evaluate how ready the data is for machine learning
    
    Args:
        data: Input DataFrame
        target_column: Optional target column name
        
    Returns:
        Dictionary with readiness scores for different aspects
    """
    readiness_scores = {}
    
    # 1. Completeness (missing values)
    missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    completeness_score = max(0, 10 - missing_percentage / 2)
    readiness_scores['completeness'] = min(10, completeness_score)
    
    # 2. Size adequacy
    row_count = len(data)
    if row_count < 100:
        size_score = 2
    elif row_count < 1000:
        size_score = 5
    elif row_count < 10000:
        size_score = 8
    else:
        size_score = 10
    readiness_scores['size_adequacy'] = size_score
    
    # 3. Feature diversity
    numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
    total_features = len(data.columns)
    
    if total_features < 3:
        diversity_score = 3
    elif total_features < 10:
        diversity_score = 6
    else:
        diversity_score = 9
    
    # Bonus for having both numeric and categorical features
    if numeric_cols > 0 and categorical_cols > 0:
        diversity_score += 1
    
    readiness_scores['feature_diversity'] = min(10, diversity_score)
    
    # 4. Data quality (duplicates, outliers)
    duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
    quality_score = max(0, 10 - duplicate_percentage)
    
    # Check for outliers in numeric columns
    outlier_columns = 0
    for col in data.select_dtypes(include=[np.number]).columns:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
            if len(outliers) > len(col_data) * 0.1:  # More than 10% outliers
                outlier_columns += 1
    
    if outlier_columns > 0:
        quality_score -= outlier_columns * 0.5
    
    readiness_scores['data_quality'] = max(0, min(10, quality_score))
    
    # 5. Target variable quality (if provided)
    if target_column and target_column in data.columns:
        target_data = data[target_column].dropna()
        target_missing_pct = (data[target_column].isnull().sum() / len(data)) * 100
        
        target_score = max(0, 10 - target_missing_pct / 2)
        
        # Check target balance for classification
        if pd.api.types.is_numeric_dtype(target_data):
            unique_ratio = target_data.nunique() / len(target_data)
            if unique_ratio < 0.1:  # Likely classification
                value_counts = target_data.value_counts()
                min_class_pct = (value_counts.min() / len(target_data)) * 100
                if min_class_pct < 5:  # Very imbalanced
                    target_score -= 2
                elif min_class_pct < 15:  # Moderately imbalanced
                    target_score -= 1
        
        readiness_scores['target_quality'] = max(0, min(10, target_score))
    else:
        readiness_scores['target_quality'] = 7  # Neutral score if no target
    
    # 6. Encoding readiness
    encoding_score = 10
    for col in data.select_dtypes(include=['object']).columns:
        unique_values = data[col].nunique()
        if unique_values > 50:  # High cardinality
            encoding_score -= 1
        elif unique_values == len(data):  # All unique (like IDs)
            encoding_score -= 2
    
    readiness_scores['encoding_readiness'] = max(0, min(10, encoding_score))
    
    return readiness_scores

def suggest_algorithms(data: pd.DataFrame, task_type: str, target_column: str = None) -> List[Dict[str, Any]]:
    """
    Suggest appropriate ML algorithms based on data characteristics
    
    Args:
        data: Input DataFrame
        task_type: Type of ML task
        target_column: Target column name
        
    Returns:
        List of algorithm recommendations with scores and explanations
    """
    algorithms = []
    
    # Data characteristics
    n_samples = len(data)
    n_features = len(data.columns) - (1 if target_column else 0)
    numeric_features = len(data.select_dtypes(include=[np.number]).columns)
    categorical_features = len(data.select_dtypes(include=['object', 'category']).columns)
    
    if task_type.lower() in ['classification', 'binary classification']:
        
        # Random Forest - Generally good for most datasets
        rf_score = 8.5
        rf_reasons = ["Handles mixed data types well", "Built-in feature importance", "Robust to outliers"]
        
        if n_samples > 1000:
            rf_score += 0.5
            rf_reasons.append("Scales well with larger datasets")
        
        if categorical_features > 0:
            rf_score += 0.5
            rf_reasons.append("Naturally handles categorical features")
        
        algorithms.append({
            'name': 'Random Forest',
            'type': 'Ensemble',
            'score': rf_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'High',
            'reasons': rf_reasons,
            'description': 'Ensemble method that combines multiple decision trees for robust predictions.',
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'accuracy_potential': 8,
            'speed_score': 7,
            'interpretability_score': 8,
            'complexity_score': 6
        })
        
        # Logistic Regression - Good baseline
        lr_score = 7.0
        lr_reasons = ["Fast training and prediction", "Provides probability estimates", "Highly interpretable"]
        
        if n_features < 20:
            lr_score += 1
            lr_reasons.append("Works well with moderate feature count")
        
        if numeric_features > categorical_features:
            lr_score += 0.5
            lr_reasons.append("Performs well with primarily numerical features")
        
        algorithms.append({
            'name': 'Logistic Regression',
            'type': 'Linear',
            'score': lr_score,
            'complexity': 'Low',
            'training_time': 'Fast',
            'interpretability': 'Very High',
            'reasons': lr_reasons,
            'description': 'Linear model for classification that provides probability estimates.',
            'hyperparameters': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'accuracy_potential': 7,
            'speed_score': 9,
            'interpretability_score': 10,
            'complexity_score': 3
        })
        
        # Gradient Boosting - High performance
        gb_score = 8.0
        gb_reasons = ["High predictive accuracy", "Handles complex patterns", "Built-in feature selection"]
        
        if n_samples > 5000:
            gb_score += 1
            gb_reasons.append("Performs exceptionally well on large datasets")
        
        algorithms.append({
            'name': 'Gradient Boosting (XGBoost)',
            'type': 'Ensemble',
            'score': gb_score,
            'complexity': 'High',
            'training_time': 'Slow',
            'interpretability': 'Medium',
            'reasons': gb_reasons,
            'description': 'Advanced boosting algorithm that often achieves state-of-the-art results.',
            'hyperparameters': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'accuracy_potential': 9,
            'speed_score': 5,
            'interpretability_score': 6,
            'complexity_score': 8
        })
        
        # SVM - Good for smaller datasets
        svm_score = 6.5
        svm_reasons = ["Effective in high dimensions", "Memory efficient"]
        
        if n_samples < 10000:
            svm_score += 1
            svm_reasons.append("Well-suited for smaller datasets")
        
        if n_features > n_samples:
            svm_score += 1
            svm_reasons.append("Handles high-dimensional data well")
        
        algorithms.append({
            'name': 'Support Vector Machine',
            'type': 'Kernel-based',
            'score': svm_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Low',
            'reasons': svm_reasons,
            'description': 'Finds optimal decision boundary using support vectors.',
            'hyperparameters': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'accuracy_potential': 7,
            'speed_score': 6,
            'interpretability_score': 4,
            'complexity_score': 7
        })
        
        # Naive Bayes - Simple and fast
        nb_score = 6.0
        nb_reasons = ["Very fast training", "Works well with small datasets", "Good baseline model"]
        
        if categorical_features > numeric_features:
            nb_score += 1
            nb_reasons.append("Performs well with categorical features")
        
        algorithms.append({
            'name': 'Naive Bayes',
            'type': 'Probabilistic',
            'score': nb_score,
            'complexity': 'Low',
            'training_time': 'Very Fast',
            'interpretability': 'High',
            'reasons': nb_reasons,
            'description': 'Simple probabilistic classifier based on feature independence assumption.',
            'hyperparameters': {
                'alpha': [0.1, 1.0, 10.0],
                'fit_prior': [True, False]
            },
            'accuracy_potential': 6,
            'speed_score': 10,
            'interpretability_score': 8,
            'complexity_score': 2
        })
    
    elif task_type.lower() == 'regression':
        
        # Random Forest Regression
        rfr_score = 8.0
        rfr_reasons = ["Handles non-linear relationships", "Robust to outliers", "No assumptions about data distribution"]
        
        if n_samples > 1000:
            rfr_score += 0.5
        
        algorithms.append({
            'name': 'Random Forest Regressor',
            'type': 'Ensemble',
            'score': rfr_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'High',
            'reasons': rfr_reasons,
            'description': 'Ensemble of decision trees for regression tasks.',
            'hyperparameters': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'accuracy_potential': 8,
            'speed_score': 7,
            'interpretability_score': 8,
            'complexity_score': 6
        })
        
        # Linear Regression
        linear_score = 7.0
        linear_reasons = ["Simple and interpretable", "Fast training", "Good baseline"]
        
        if n_features < 20:
            linear_score += 1
            linear_reasons.append("Works well with moderate feature count")
        
        algorithms.append({
            'name': 'Linear Regression',
            'type': 'Linear',
            'score': linear_score,
            'complexity': 'Low',
            'training_time': 'Very Fast',
            'interpretability': 'Very High',
            'reasons': linear_reasons,
            'description': 'Linear relationship modeling between features and target.',
            'hyperparameters': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'accuracy_potential': 6,
            'speed_score': 10,
            'interpretability_score': 10,
            'complexity_score': 2
        })
        
        # Gradient Boosting Regression
        gbr_score = 8.5
        gbr_reasons = ["High accuracy", "Handles complex patterns", "Feature importance"]
        
        algorithms.append({
            'name': 'Gradient Boosting Regressor',
            'type': 'Ensemble',
            'score': gbr_score,
            'complexity': 'High',
            'training_time': 'Slow',
            'interpretability': 'Medium',
            'reasons': gbr_reasons,
            'description': 'Sequential ensemble method that builds models to correct previous errors.',
            'hyperparameters': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'accuracy_potential': 9,
            'speed_score': 5,
            'interpretability_score': 6,
            'complexity_score': 8
        })
        
        # SVR
        svr_score = 6.5
        svr_reasons = ["Effective in high dimensions", "Robust to outliers"]
        
        if n_samples < 10000:
            svr_score += 1
        
        algorithms.append({
            'name': 'Support Vector Regression',
            'type': 'Kernel-based',
            'score': svr_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Low',
            'reasons': svr_reasons,
            'description': 'SVM adapted for regression tasks using epsilon-insensitive loss.',
            'hyperparameters': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'accuracy_potential': 7,
            'speed_score': 6,
            'interpretability_score': 4,
            'complexity_score': 7
        })
    
    elif task_type.lower() == 'clustering':
        
        # K-Means
        kmeans_score = 8.0
        kmeans_reasons = ["Simple and efficient", "Works well with spherical clusters", "Scalable"]
        
        algorithms.append({
            'name': 'K-Means',
            'type': 'Centroid-based',
            'score': kmeans_score,
            'complexity': 'Low',
            'training_time': 'Fast',
            'interpretability': 'High',
            'reasons': kmeans_reasons,
            'description': 'Partitions data into k clusters by minimizing within-cluster sum of squares.',
            'hyperparameters': {
                'n_clusters': [2, 3, 5, 8, 10],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20]
            },
            'accuracy_potential': 7,
            'speed_score': 9,
            'interpretability_score': 9,
            'complexity_score': 3
        })
        
        # DBSCAN
        dbscan_score = 7.5
        dbscan_reasons = ["Finds clusters of arbitrary shape", "Robust to outliers", "No need to specify cluster count"]
        
        algorithms.append({
            'name': 'DBSCAN',
            'type': 'Density-based',
            'score': dbscan_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'reasons': dbscan_reasons,
            'description': 'Groups together points in high-density areas and marks outliers.',
            'hyperparameters': {
                'eps': [0.3, 0.5, 0.7, 1.0],
                'min_samples': [3, 5, 10, 15]
            },
            'accuracy_potential': 8,
            'speed_score': 7,
            'interpretability_score': 7,
            'complexity_score': 6
        })

    elif task_type.lower() in ['anomaly detection', 'anomaly-detection']:
        # Isolation Forest
        if_score = 8.0
        if_reasons = [
            "Effective for high-dimensional data",
            "Scales well",
            "Works without labeled anomalies"
        ]
        algorithms.append({
            'name': 'Isolation Forest',
            'type': 'Tree-based',
            'score': if_score,
            'complexity': 'Low',
            'training_time': 'Fast',
            'interpretability': 'Medium',
            'reasons': if_reasons,
            'description': 'Detects anomalies by isolating observations.',
            'hyperparameters': {
                'n_estimators': [100, 200],
                'contamination': [0.01, 0.05, 0.1]
            },
            'accuracy_potential': 7,
            'speed_score': 9,
            'interpretability_score': 6,
            'complexity_score': 3
        })

        # Local Outlier Factor
        lof_score = 7.5
        lof_reasons = [
            "Captures local density variations",
            "Good for complex cluster structures"
        ]
        algorithms.append({
            'name': 'Local Outlier Factor',
            'type': 'Density-based',
            'score': lof_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Low',
            'reasons': lof_reasons,
            'description': 'Identifies anomalies by comparing local density.',
            'hyperparameters': {
                'n_neighbors': [20, 35, 50],
                'contamination': [0.01, 0.05, 0.1]
            },
            'accuracy_potential': 7,
            'speed_score': 6,
            'interpretability_score': 4,
            'complexity_score': 6
        })

    elif task_type.lower() in ['time series', 'time-series', 'timeseries']:
        # Time series suggestions (supervised with lag features)
        gbr_ts_score = 7.5
        gbr_ts_reasons = [
            "Flexible non-linear modeling",
            "Works with lag features",
            "Does not require strict stationarity"
        ]
        algorithms.append({
            'name': 'Gradient Boosting (with lag features)',
            'type': 'Ensemble',
            'score': gbr_ts_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'reasons': gbr_ts_reasons,
            'description': 'Supervised model using generated lag/time features.',
            'hyperparameters': {
                'n_estimators': [200, 500],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'accuracy_potential': 8,
            'speed_score': 6,
            'interpretability_score': 6,
            'complexity_score': 6
        })

        rf_ts_score = 7.0
        rf_ts_reasons = [
            "Robust to noise",
            "Captures non-linearities",
            "Less sensitive to feature scaling"
        ]
        algorithms.append({
            'name': 'Random Forest (with lag features)',
            'type': 'Ensemble',
            'score': rf_ts_score,
            'complexity': 'Medium',
            'training_time': 'Medium',
            'interpretability': 'Medium',
            'reasons': rf_ts_reasons,
            'description': 'Tree ensemble using engineered lag features.',
            'hyperparameters': {
                'n_estimators': [200, 500],
                'max_depth': [10, 20, None]
            },
            'accuracy_potential': 7,
            'speed_score': 6,
            'interpretability_score': 6,
            'complexity_score': 6
        })
    
    # Sort algorithms by score (descending)
    algorithms.sort(key=lambda x: x['score'], reverse=True)
    
    return algorithms

def get_ml_recommendations(data: pd.DataFrame, task_type: str = None, target_column: str = None) -> Dict[str, Any]:
    """
    Get comprehensive ML recommendations for a dataset
    
    Args:
        data: Input DataFrame
        task_type: Type of ML task
        target_column: Target column name
        
    Returns:
        Dictionary containing comprehensive ML recommendations
    """
    recommendations = {}
    
    # Determine task type if not provided
    if task_type is None or task_type == "Auto-Detect":
        if target_column:
            task_type = analyze_target_variable(data, target_column)
        else:
            task_type = "Clustering"  # Default to unsupervised if no target
    
    recommendations['task_type'] = task_type
    
    # Evaluate data readiness
    readiness_scores = evaluate_data_readiness(data, target_column)
    recommendations['data_readiness'] = readiness_scores
    recommendations['data_readiness_score'] = np.mean(list(readiness_scores.values()))
    
    # Get algorithm recommendations
    algorithms = suggest_algorithms(data, task_type, target_column)
    recommendations['recommended_algorithms'] = algorithms
    
    # Feature engineering suggestions
    feature_suggestions = []
    
    # Missing value handling
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        feature_suggestions.append({
            'technique': 'Missing Value Imputation',
            'reason': f'{len(missing_cols)} columns have missing values',
            'impact': 'High',
            'columns': missing_cols
        })
    
    # Categorical encoding
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        feature_suggestions.append({
            'technique': 'Categorical Encoding',
            'reason': f'{len(categorical_cols)} categorical columns need encoding',
            'impact': 'High',
            'columns': categorical_cols
        })
    
    # Feature scaling
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) > 1:
        # Check if scaling is needed
        ranges = []
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            col_data = data[col].dropna()
            if len(col_data) > 0:
                ranges.append(col_data.max() - col_data.min())
        
        if ranges and max(ranges) / min(ranges) > 100:  # Large scale differences
            feature_suggestions.append({
                'technique': 'Feature Scaling',
                'reason': 'Features have very different scales',
                'impact': 'Medium',
                'columns': numeric_cols
            })
    
    # Outlier handling
    outlier_cols = []
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
            if len(outliers) > len(col_data) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
    
    if outlier_cols:
        feature_suggestions.append({
            'technique': 'Outlier Treatment',
            'reason': f'{len(outlier_cols)} columns have significant outliers',
            'impact': 'Medium',
            'columns': outlier_cols
        })
    
    recommendations['feature_engineering'] = feature_suggestions
    
    # Preprocessing steps
    preprocessing_steps = []
    
    # Data cleaning
    if data.duplicated().sum() > 0:
        preprocessing_steps.append({
            'step': 'Remove Duplicates',
            'description': f'Remove {data.duplicated().sum()} duplicate rows',
            'priority': 'High',
            'columns': list(data.columns)
        })
    
    # Missing value handling
    if missing_cols:
        preprocessing_steps.append({
            'step': 'Handle Missing Values',
            'description': 'Impute or remove missing values',
            'priority': 'High',
            'columns': missing_cols
        })
    
    # Feature encoding
    if categorical_cols:
        preprocessing_steps.append({
            'step': 'Encode Categorical Variables',
            'description': 'Convert categorical variables to numerical format',
            'priority': 'High',
            'columns': categorical_cols
        })
    
    # Feature scaling
    if task_type.lower() in ['classification', 'regression'] and len(numeric_cols) > 1:
        preprocessing_steps.append({
            'step': 'Scale Features',
            'description': 'Normalize or standardize numerical features',
            'priority': 'Medium',
            'columns': numeric_cols
        })
    
    recommendations['preprocessing_steps'] = preprocessing_steps
    
    # Evaluation strategy
    evaluation = {}
    
    if task_type.lower() in ['classification', 'binary classification']:
        evaluation['metrics'] = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
        evaluation['validation_method'] = 'Stratified K-Fold Cross-Validation'
        evaluation['cv_folds'] = 5
        evaluation['test_size'] = 0.2
    
    elif task_type.lower() == 'regression':
        evaluation['metrics'] = ['MAE', 'MSE', 'RMSE', 'R-squared']
        evaluation['validation_method'] = 'K-Fold Cross-Validation'
        evaluation['cv_folds'] = 5
        evaluation['test_size'] = 0.2
    
    elif task_type.lower() == 'clustering':
        evaluation['metrics'] = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
        evaluation['validation_method'] = 'Internal Validation'

    elif task_type.lower() in ['anomaly detection', 'anomaly-detection']:
        evaluation['metrics'] = ['Precision@k (if labels)', 'Recall@k (if labels)', 'PR-AUC (if labels)']
        evaluation['validation_method'] = 'Unsupervised Evaluation / Manual Inspection'

    elif task_type.lower() in ['time series', 'time-series', 'timeseries']:
        evaluation['metrics'] = ['MAE', 'RMSE', 'MAPE']
        evaluation['validation_method'] = 'Temporal Train/Test Split'
        evaluation['test_size'] = 0.2
    
    recommendations['evaluation_strategy'] = evaluation
    
    # Leakage diagnostics (simple heuristics)
    leakage_warnings: List[str] = []
    if target_column and target_column in data.columns:
        lower_target = target_column.lower()
        leak_like_cols = [
            c for c in data.columns if c != target_column and (
                lower_target in c.lower() or 'leak' in c.lower() or 'target' in c.lower()
            )
        ]
        if leak_like_cols:
            leakage_warnings.append(
                f"Columns suspicious for leakage relative to target: {', '.join(leak_like_cols[:10])}"
            )
        # High correlation with numeric target
        if pd.api.types.is_numeric_dtype(data[target_column]):
            try:
                corr = data.select_dtypes(include=[np.number]).corr()[target_column].abs().sort_values(ascending=False)
                high_corr = corr[(corr.index != target_column) & (corr > 0.95)].index.tolist()
                if high_corr:
                    leakage_warnings.append(
                        f"Features highly correlated with target (>0.95): {', '.join(high_corr[:10])}"
                    )
            except Exception:
                pass
    if leakage_warnings:
        recommendations['leakage_warnings'] = leakage_warnings

    # Performance tips
    performance_tips = []
    
    if len(data) > 100000:
        performance_tips.append("Consider using sampling for initial model development")
        performance_tips.append("Use algorithms that scale well with large datasets")
    
    if len(data.columns) > 50:
        performance_tips.append("Consider feature selection to reduce dimensionality")
        performance_tips.append("Use regularization to prevent overfitting")
    
    if readiness_scores['data_quality'] < 7:
        performance_tips.append("Focus on data cleaning before model building")
    
    recommendations['performance_tips'] = performance_tips
    
    # Potential challenges
    challenges = []
    
    if readiness_scores['completeness'] < 7:
        challenges.append("High missing data rate may impact model performance")
    
    if readiness_scores['size_adequacy'] < 5:
        challenges.append("Small dataset size may limit model complexity")
    
    if task_type.lower() in ['classification', 'binary classification'] and target_column:
        # Check class balance
        value_counts = data[target_column].value_counts()
        if len(value_counts) > 1:
            min_class_pct = (value_counts.min() / len(data)) * 100
            if min_class_pct < 10:
                challenges.append("Imbalanced target classes may require special handling")
    
    recommendations['challenges'] = challenges
    
    return recommendations

def generate_model_code(algorithm: Dict[str, Any], task_type: str, target_column: str = None) -> str:
    """
    Generate Python code for implementing the recommended algorithm
    
    Args:
        algorithm: Algorithm dictionary from suggestions
        task_type: ML task type
        target_column: Target column name
        
    Returns:
        String containing Python code
    """
    algo_name = algorithm['name']
    hyperparams = algorithm.get('hyperparameters', {})
    
    code = f"""
# {algo_name} Implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
"""
    
    # Import specific algorithm
    if 'Random Forest' in algo_name:
        if task_type.lower() == 'regression':
            code += "from sklearn.ensemble import RandomForestRegressor\n"
            model_class = "RandomForestRegressor"
        else:
            code += "from sklearn.ensemble import RandomForestClassifier\n"
            model_class = "RandomForestClassifier"
    
    elif 'Logistic Regression' in algo_name:
        code += "from sklearn.linear_model import LogisticRegression\n"
        model_class = "LogisticRegression"
    
    elif 'Gradient Boosting' in algo_name or 'XGBoost' in algo_name:
        if task_type.lower() == 'regression':
            code += "from sklearn.ensemble import GradientBoostingRegressor\n"
            model_class = "GradientBoostingRegressor"
        else:
            code += "from sklearn.ensemble import GradientBoostingClassifier\n"
            model_class = "GradientBoostingClassifier"
    
    elif 'Support Vector' in algo_name:
        if task_type.lower() == 'regression':
            code += "from sklearn.svm import SVR\n"
            model_class = "SVR"
        else:
            code += "from sklearn.svm import SVC\n"
            model_class = "SVC"
    
    elif 'Linear Regression' in algo_name:
        code += "from sklearn.linear_model import LinearRegression\n"
        model_class = "LinearRegression"
    
    elif 'Naive Bayes' in algo_name:
        code += "from sklearn.naive_bayes import GaussianNB\n"
        model_class = "GaussianNB"
    
    elif 'K-Means' in algo_name:
        code += "from sklearn.cluster import KMeans\n"
        model_class = "KMeans"
    
    elif 'DBSCAN' in algo_name:
        code += "from sklearn.cluster import DBSCAN\n"
        model_class = "DBSCAN"
    
    elif 'Isolation Forest' in algo_name:
        code += "from sklearn.ensemble import IsolationForest\n"
        model_class = "IsolationForest"
    elif 'Local Outlier Factor' in algo_name:
        code += "from sklearn.neighbors import LocalOutlierFactor\n"
        model_class = "LocalOutlierFactor"
    
    else:
        model_class = "YourModel"
    
    code += f"""
# Load and prepare your data
# data = pd.read_csv('your_data.csv')

# Preprocessing
# Handle missing values
# data = data.fillna(data.mean())  # or use more sophisticated imputation

# Encode categorical variables if needed
# le = LabelEncoder()
# for col in data.select_dtypes(include=['object']).columns:
#     data[col] = le.fit_transform(data[col])
"""
    
    if task_type.lower() in ['classification', 'regression']:
        code += f"""
# Prepare features and target
X = data.drop('{target_column}', axis=1)  # Features
y = data['{target_column}']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42{', stratify=y' if task_type.lower() in ['classification', 'binary classification'] else ''}
)

# Scale features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize model with recommended hyperparameters
model = {model_class}(
"""
        
        # Add hyperparameters
        if hyperparams:
            for param, values in hyperparams.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], str):
                        code += f"    {param}='{values[0]}',\n"
                    else:
                        code += f"    {param}={values[0]},\n"
        
        code += """    random_state=42
)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
"""
        
        if task_type.lower() in ['classification', 'binary classification']:
            code += """print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
"""
        else:  # Regression
            code += """print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.3f}")

# Cross-validation score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-validation R2: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
"""
    
    elif task_type.lower() == 'clustering':
        code += f"""
# Prepare features (no target for clustering)
X = data.select_dtypes(include=[np.number])  # Use only numeric features

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize clustering model
model = {model_class}(
"""
        
        if hyperparams:
            for param, values in hyperparams.items():
                if isinstance(values, list) and len(values) > 0:
                    code += f"    {param}={values[0]},\n"
        
        code += """)

# Fit the model and get cluster labels
cluster_labels = model.fit_predict(X_scaled)

# Add cluster labels to original data
data['cluster'] = cluster_labels

# Evaluate clustering (if applicable)
from sklearn.metrics import silhouette_score
if len(set(cluster_labels)) > 1:
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Average silhouette score: {{silhouette_avg:.3f}}")

# Analyze clusters
print("Cluster summary:")
print(data.groupby('cluster').describe())
"""
    
    elif task_type.lower() in ['anomaly detection', 'anomaly-detection']:
        code += f"""
# Prepare features (unsupervised anomaly detection)
X = data.select_dtypes(include=[np.number]).copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize anomaly detection model
model = {model_class}(
"""
        if hyperparams:
            for param, values in hyperparams.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], str):
                        code += f"    {param}='{values[0]}',\n"
                    else:
                        code += f"    {param}={values[0]},\n"
        code += ")\n\n"
        code += """
# Fit model and get anomaly scores/labels
if hasattr(model, 'fit_predict') and model.__class__.__name__ == 'LocalOutlierFactor':
    labels = model.fit_predict(X_scaled)
else:
    model.fit(X_scaled)
    # For IsolationForest: predict returns 1 for normal, -1 for anomaly
    labels = model.predict(X_scaled)

# Report anomaly counts
import numpy as np
anomaly_count = int(np.sum(labels == -1))
normal_count = int(np.sum(labels == 1))
print(f"Detected anomalies: {anomaly_count} | Normal: {normal_count}")
"""

    elif task_type.lower() in ['time series', 'time-series', 'timeseries']:
        code += f"""
# Time Series Forecasting with lag features
from sklearn.ensemble import GradientBoostingRegressor

if '{target_column}' == 'None' or '{target_column}' is None:
    raise ValueError('A target column is required for time series tasks.')

df = data.copy()

# Try to infer datetime column
datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
if not datetime_cols:
    # Attempt to parse a likely date column
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c])
            df[c] = parsed
            datetime_cols = [c]
            break
        except Exception:
            continue
if not datetime_cols:
    raise ValueError('No datetime-like column found. Ensure your data has a date/time column.')

date_col = datetime_cols[0]
df = df.sort_values(by=date_col).reset_index(drop=True)

y = df['{target_column}'].astype(float)

# Create lag features
max_lag = 12
for lag in range(1, max_lag + 1):
    df[f'lag_{{lag}}'] = y.shift(lag)

feature_cols = [c for c in df.columns if c.startswith('lag_')]
ts = df[[date_col] + feature_cols].copy()
ts['y'] = y
ts = ts.dropna().reset_index(drop=True)

X = ts[feature_cols]
y = ts['y']

# Temporal split (last 20% as test)
split_idx = int(len(ts) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = (np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-8, None))).mean() * 100
print(f"MAE: {{mae:.3f}} | RMSE: {{rmse:.3f}} | MAPE: {{mape:.2f}}%")
"""

    code += """
# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
"""
    
    return code

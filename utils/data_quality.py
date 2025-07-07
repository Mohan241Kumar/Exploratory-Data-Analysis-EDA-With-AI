import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def detect_missing_data(data):
    """
    Detect missing data patterns
    """
    missing_info = {}
    
    # Column-wise missing data
    missing_counts = data.isnull().sum()
    missing_percentages = (missing_counts / len(data)) * 100
    
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    missing_info = {
        'count': missing_counts.sum(),
        'severity': 'High' if missing_percentages.max() > 50 else 'Medium' if missing_percentages.max() > 10 else 'Low',
        'description': f'{len(columns_with_missing)} columns have missing values',
        'affected_columns': columns_with_missing,
        'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
    }
    
    return missing_info

def detect_data_type_mismatches(data):
    """
    Detect potential data type mismatches
    """
    mismatches = []
    
    for col in data.columns:
        if data[col].dtype == 'object':
            # Check if it should be numeric
            try:
                numeric_conversion = pd.to_numeric(data[col], errors='coerce')
                conversion_rate = numeric_conversion.notna().sum() / len(data)
                
                if conversion_rate > 0.8:  # 80% can be converted
                    mismatches.append({
                        'column': col,
                        'current_type': 'object',
                        'suggested_type': 'numeric',
                        'conversion_rate': conversion_rate
                    })
            except:
                pass
            
            # Check if it should be datetime
            try:
                datetime_conversion = pd.to_datetime(data[col], errors='coerce')
                conversion_rate = datetime_conversion.notna().sum() / len(data)
                
                if conversion_rate > 0.8:  # 80% can be converted
                    mismatches.append({
                        'column': col,
                        'current_type': 'object',
                        'suggested_type': 'datetime',
                        'conversion_rate': conversion_rate
                    })
            except:
                pass
    
    return {
        'count': len(mismatches),
        'severity': 'High' if len(mismatches) > 5 else 'Medium' if len(mismatches) > 0 else 'Low',
        'description': f'{len(mismatches)} columns may have incorrect data types',
        'mismatches': mismatches
    }

def detect_outliers(data):
    """
    Detect outliers using IQR method
    """
    outlier_info = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    total_outliers = 0
    outlier_columns = []
    
    for col in numeric_cols:
        col_data = data[col].dropna()
        
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            if len(outliers) > 0:
                outlier_columns.append({
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(col_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
                total_outliers += len(outliers)
    
    return {
        'count': total_outliers,
        'severity': 'High' if total_outliers > len(data) * 0.1 else 'Medium' if total_outliers > 0 else 'Low',
        'description': f'{total_outliers} outliers detected across {len(outlier_columns)} columns',
        'outlier_columns': outlier_columns
    }

def detect_duplicate_rows(data):
    """
    Detect duplicate rows
    """
    duplicate_count = data.duplicated().sum()
    
    return {
        'count': duplicate_count,
        'severity': 'High' if duplicate_count > len(data) * 0.1 else 'Medium' if duplicate_count > 0 else 'Low',
        'description': f'{duplicate_count} duplicate rows found ({duplicate_count/len(data)*100:.1f}%)',
        'duplicate_percentage': (duplicate_count / len(data)) * 100
    }

def detect_inconsistent_categories(data):
    """
    Detect inconsistent categorical values
    """
    inconsistencies = []
    cat_cols = data.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        unique_vals = data[col].dropna().unique()
        
        # Check for case variations
        case_groups = {}
        for val in unique_vals:
            if isinstance(val, str):
                lower_val = val.lower().strip()
                if lower_val not in case_groups:
                    case_groups[lower_val] = []
                case_groups[lower_val].append(val)
        
        # Find groups with multiple variations
        for lower_val, variations in case_groups.items():
            if len(variations) > 1:
                inconsistencies.append({
                    'column': col,
                    'base_value': lower_val,
                    'variations': variations,
                    'variation_count': len(variations)
                })
    
    return {
        'count': len(inconsistencies),
        'severity': 'Medium' if len(inconsistencies) > 0 else 'Low',
        'description': f'{len(inconsistencies)} inconsistent categories detected',
        'inconsistencies': inconsistencies
    }

def detect_class_imbalance(data):
    """
    Detect class imbalance in categorical columns
    """
    imbalanced_cols = []
    cat_cols = data.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        value_counts = data[col].value_counts()
        
        if len(value_counts) > 1:
            # Calculate imbalance ratio
            majority_class = value_counts.max()
            minority_class = value_counts.min()
            imbalance_ratio = majority_class / minority_class
            
            if imbalance_ratio > 3:  # Significant imbalance
                imbalanced_cols.append({
                    'column': col,
                    'imbalance_ratio': imbalance_ratio,
                    'majority_class_count': majority_class,
                    'minority_class_count': minority_class,
                    'classes': len(value_counts)
                })
    
    return {
        'count': len(imbalanced_cols),
        'severity': 'High' if any(col['imbalance_ratio'] > 10 for col in imbalanced_cols) else 'Medium' if len(imbalanced_cols) > 0 else 'Low',
        'description': f'{len(imbalanced_cols)} columns show class imbalance',
        'imbalanced_columns': imbalanced_cols
    }

def detect_skewed_distributions(data):
    """
    Detect skewed distributions in numerical columns
    """
    skewed_cols = []
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = data[col].dropna()
        
        if len(col_data) > 0:
            skewness = stats.skew(col_data)
            
            if abs(skewness) > 1:  # Highly skewed
                skewed_cols.append({
                    'column': col,
                    'skewness': skewness,
                    'interpretation': 'Highly Skewed' if abs(skewness) > 2 else 'Moderately Skewed'
                })
    
    return {
        'count': len(skewed_cols),
        'severity': 'Medium' if len(skewed_cols) > 0 else 'Low',
        'description': f'{len(skewed_cols)} columns have skewed distributions',
        'skewed_columns': skewed_cols
    }

def detect_low_variance_columns(data):
    """
    Detect columns with low variance
    """
    low_variance_cols = []
    
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            # For numeric columns, check variance
            variance = data[col].var()
            if variance < 0.01:  # Very low variance
                low_variance_cols.append({
                    'column': col,
                    'variance': variance,
                    'type': 'numeric'
                })
        else:
            # For categorical columns, check unique value ratio
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.01:  # Less than 1% unique values
                low_variance_cols.append({
                    'column': col,
                    'unique_ratio': unique_ratio,
                    'type': 'categorical'
                })
    
    return {
        'count': len(low_variance_cols),
        'severity': 'Medium' if len(low_variance_cols) > 0 else 'Low',
        'description': f'{len(low_variance_cols)} columns have low variance',
        'low_variance_columns': low_variance_cols
    }

def detect_multicollinearity(data):
    """
    Detect multicollinearity among numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return {
            'count': 0,
            'severity': 'Low',
            'description': 'Not enough numerical columns for multicollinearity analysis',
            'correlated_pairs': []
        }
    
    corr_matrix = data[numeric_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) > 0.8:  # High correlation
                high_corr_pairs.append({
                    'column_1': corr_matrix.columns[i],
                    'column_2': corr_matrix.columns[j],
                    'correlation': corr_val,
                    'strength': 'Very High' if abs(corr_val) > 0.9 else 'High'
                })
    
    return {
        'count': len(high_corr_pairs),
        'severity': 'High' if len(high_corr_pairs) > 0 else 'Low',
        'description': f'{len(high_corr_pairs)} highly correlated pairs detected',
        'correlated_pairs': high_corr_pairs
    }

def detect_invalid_values(data):
    """
    Detect invalid or impossible values based on domain knowledge
    """
    invalid_values = []
    
    for col in data.columns:
        col_data = data[col].dropna()
        
        # Age columns
        if 'age' in col.lower():
            invalid_ages = col_data[(col_data < 0) | (col_data > 150)]
            if len(invalid_ages) > 0:
                invalid_values.append({
                    'column': col,
                    'issue': 'Invalid Age',
                    'invalid_count': len(invalid_ages),
                    'examples': invalid_ages.head(5).tolist()
                })
        
        # Percentage columns
        if any(keyword in col.lower() for keyword in ['percent', 'rate', 'ratio']):
            invalid_percentages = col_data[(col_data < 0) | (col_data > 100)]
            if len(invalid_percentages) > 0:
                invalid_values.append({
                    'column': col,
                    'issue': 'Invalid Percentage',
                    'invalid_count': len(invalid_percentages),
                    'examples': invalid_percentages.head(5).tolist()
                })
        
        # Salary/Price columns
        if any(keyword in col.lower() for keyword in ['salary', 'price', 'cost', 'amount']):
            invalid_values_neg = col_data[col_data < 0]
            if len(invalid_values_neg) > 0:
                invalid_values.append({
                    'column': col,
                    'issue': 'Negative Financial Value',
                    'invalid_count': len(invalid_values_neg),
                    'examples': invalid_values_neg.head(5).tolist()
                })
    
    return {
        'count': len(invalid_values),
        'severity': 'High' if len(invalid_values) > 0 else 'Low',
        'description': f'{len(invalid_values)} columns contain invalid values',
        'invalid_values': invalid_values
    }

def detect_all_issues(data):
    """
    Detect all common data quality issues
    """
    issues = {
        'missing_data': detect_missing_data(data),
        'data_type_mismatches': detect_data_type_mismatches(data),
        'outliers': detect_outliers(data),
        'duplicate_rows': detect_duplicate_rows(data),
        'inconsistent_categories': detect_inconsistent_categories(data),
        'class_imbalance': detect_class_imbalance(data),
        'skewed_distributions': detect_skewed_distributions(data),
        'low_variance_columns': detect_low_variance_columns(data),
        'multicollinearity': detect_multicollinearity(data),
        'invalid_values': detect_invalid_values(data)
    }
    
    return issues

def get_detailed_issue_analysis(data, issue_type):
    """
    Get detailed analysis for a specific issue type
    """
    if issue_type == 'missing_data':
        return detect_missing_data(data)
    elif issue_type == 'data_type_mismatches':
        return detect_data_type_mismatches(data)
    elif issue_type == 'outliers':
        return detect_outliers(data)
    elif issue_type == 'duplicate_rows':
        return detect_duplicate_rows(data)
    elif issue_type == 'inconsistent_categories':
        return detect_inconsistent_categories(data)
    elif issue_type == 'class_imbalance':
        return detect_class_imbalance(data)
    elif issue_type == 'skewed_distributions':
        return detect_skewed_distributions(data)
    elif issue_type == 'low_variance_columns':
        return detect_low_variance_columns(data)
    elif issue_type == 'multicollinearity':
        return detect_multicollinearity(data)
    elif issue_type == 'invalid_values':
        return detect_invalid_values(data)
    else:
        return {}

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def load_data(uploaded_file):
    """
    Load CSV data from uploaded file
    """
    try:
        # Read CSV file
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_basic_info(data):
    """
    Get basic information about the dataset
    """
    info = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'duplicate_rows': data.duplicated().sum()
    }
    return info

def detect_column_types(data):
    """
    Detect and suggest appropriate data types for columns
    """
    suggestions = {}
    
    for col in data.columns:
        current_type = str(data[col].dtype)
        suggestions[col] = {
            'current_type': current_type,
            'suggested_type': current_type,
            'reason': 'No change needed'
        }
        
        if data[col].dtype == 'object':
            # Check if it could be numeric
            try:
                numeric_conversion = pd.to_numeric(data[col], errors='coerce')
                non_null_ratio = numeric_conversion.notna().sum() / len(data)
                
                if non_null_ratio > 0.8:  # 80% can be converted
                    suggestions[col]['suggested_type'] = 'numeric'
                    suggestions[col]['reason'] = f'{non_null_ratio:.1%} values can be converted to numeric'
                    continue
            except:
                pass
            
            # Check if it could be datetime
            try:
                datetime_conversion = pd.to_datetime(data[col], errors='coerce')
                non_null_ratio = datetime_conversion.notna().sum() / len(data)
                
                if non_null_ratio > 0.8:  # 80% can be converted
                    suggestions[col]['suggested_type'] = 'datetime'
                    suggestions[col]['reason'] = f'{non_null_ratio:.1%} values can be converted to datetime'
                    continue
            except:
                pass
            
            # Check if it could be categorical
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.1:  # Less than 10% unique values
                suggestions[col]['suggested_type'] = 'category'
                suggestions[col]['reason'] = f'Only {unique_ratio:.1%} unique values - good candidate for categorical'
    
    return suggestions

def calculate_data_quality_score(data):
    """
    Calculate overall data quality score
    """
    scores = {}
    
    # Completeness (0-100)
    total_cells = data.shape[0] * data.shape[1]
    missing_cells = data.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    scores['completeness'] = completeness
    
    # Uniqueness (0-100)
    duplicate_rows = data.duplicated().sum()
    uniqueness = ((len(data) - duplicate_rows) / len(data)) * 100
    scores['uniqueness'] = uniqueness
    
    # Consistency (basic check for categorical columns)
    consistency_issues = 0
    cat_cols = data.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        unique_vals = data[col].dropna().unique()
        # Check for potential case inconsistencies
        if len(unique_vals) > 1:
            lower_vals = set(str(val).lower() for val in unique_vals if isinstance(val, str))
            if len(lower_vals) < len(unique_vals):
                consistency_issues += 1
    
    consistency = max(0, 100 - (consistency_issues / len(cat_cols) * 100)) if len(cat_cols) > 0 else 100
    scores['consistency'] = consistency
    
    # Validity (domain-specific checks)
    validity_issues = 0
    total_checks = 0
    
    for col in data.columns:
        if 'age' in col.lower():
            total_checks += 1
            invalid_ages = data[col][(data[col] < 0) | (data[col] > 150)].count()
            if invalid_ages > 0:
                validity_issues += 1
        
        if any(keyword in col.lower() for keyword in ['percent', 'rate', 'ratio']):
            total_checks += 1
            invalid_percentages = data[col][(data[col] < 0) | (data[col] > 100)].count()
            if invalid_percentages > 0:
                validity_issues += 1
    
    validity = max(0, 100 - (validity_issues / total_checks * 100)) if total_checks > 0 else 100
    scores['validity'] = validity
    
    # Overall score (weighted average)
    overall_score = (completeness * 0.3 + uniqueness * 0.3 + consistency * 0.2 + validity * 0.2)
    scores['overall'] = overall_score
    
    return scores

def generate_summary_statistics(data):
    """
    Generate comprehensive summary statistics
    """
    summary = {}
    
    # Basic info
    summary['basic_info'] = {
        'rows': data.shape[0],
        'columns': data.shape[1],
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    # Numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = data[numeric_cols].describe().to_dict()
    
    # Categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_summary = {}
        for col in cat_cols:
            cat_summary[col] = {
                'unique_values': data[col].nunique(),
                'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                'frequency': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
            }
        summary['categorical_summary'] = cat_summary
    
    # Data types
    summary['data_types'] = data.dtypes.value_counts().to_dict()
    
    return summary

def create_data_profile(data):
    """
    Create a comprehensive data profile
    """
    profile = {
        'basic_info': get_basic_info(data),
        'type_suggestions': detect_column_types(data),
        'quality_scores': calculate_data_quality_score(data),
        'summary_statistics': generate_summary_statistics(data)
    }
    
    return profile

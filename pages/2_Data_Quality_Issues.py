import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from scipy import stats
from utils.data_quality import detect_all_issues, get_detailed_issue_analysis
from utils.visualizations import create_missing_data_viz, create_outlier_viz, create_correlation_heatmap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Data Quality Issues",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Data Quality Issues Analysis")
st.markdown("Comprehensive analysis of 10 common data quality problems")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

# Detect all issues
if 'data_issues' not in st.session_state:
    st.session_state.data_issues = detect_all_issues(data)

data_issues = st.session_state.data_issues

# Issue selection
st.header("🎯 Select Issue to Analyze")

issue_options = {
    'missing_data': '1. Missing Data',
    'data_type_mismatches': '2. Data Type Mismatches',
    'outliers': '3. Outliers',
    'duplicate_rows': '4. Duplicate Rows',
    'inconsistent_categories': '5. Inconsistent Categories',
    'class_imbalance': '6. Class Imbalance',
    'skewed_distributions': '7. Skewed Distributions',
    'low_variance_columns': '8. Low Variance Columns',
    'multicollinearity': '9. Multicollinearity',
    'invalid_values': '10. Invalid/Impossible Values'
}

selected_issue = st.selectbox(
    "Choose an issue to analyze:",
    options=list(issue_options.keys()),
    format_func=lambda x: issue_options[x]
)

# Display issue analysis
st.header(f"📊 {issue_options[selected_issue]} Analysis")

if selected_issue == 'missing_data':
    st.subheader("Missing Data Analysis")
    
    missing_summary = data.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Data Summary**")
            missing_df = pd.DataFrame({
                'Column': missing_summary.index,
                'Missing Count': missing_summary.values,
                'Missing Percentage': (missing_summary.values / len(data) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Missing data visualization
            fig = px.bar(
                missing_df, 
                x='Missing Percentage', 
                y='Column',
                orientation='h',
                title='Missing Data by Column'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing data heatmap
        st.subheader("Missing Data Pattern")
        fig, ax = plt.subplots(figsize=(12, 8))
        msno.heatmap(data, ax=ax)
        st.pyplot(fig)
        
        # Missing data matrix
        st.subheader("Missing Data Matrix")
        fig, ax = plt.subplots(figsize=(12, 8))
        msno.matrix(data, ax=ax)
        st.pyplot(fig)
        
    else:
        st.success("🎉 No missing data found!")

elif selected_issue == 'data_type_mismatches':
    st.subheader("Data Type Analysis")
    
    # Display current data types
    dtype_df = pd.DataFrame({
        'Column': data.dtypes.index,
        'Current Type': data.dtypes.values.astype(str),
        'Non-Null Count': data.count().values,
        'Sample Values': [str(data[col].dropna().head(3).tolist()) for col in data.columns]
    })
    
    st.dataframe(dtype_df, use_container_width=True)
    
    # Suggestions for type conversion
    st.subheader("Type Conversion Suggestions")
    suggestions = []
    
    for col in data.columns:
        if data[col].dtype == 'object':
            # Check if it could be numeric
            try:
                pd.to_numeric(data[col], errors='coerce')
                non_null_after = pd.to_numeric(data[col], errors='coerce').notna().sum()
                if non_null_after > 0:
                    suggestions.append({
                        'Column': col,
                        'Current Type': 'object',
                        'Suggested Type': 'numeric',
                        'Reason': f'Can convert {non_null_after} values to numeric'
                    })
            except:
                pass
            
            # Check if it could be datetime
            try:
                pd.to_datetime(data[col], errors='coerce')
                non_null_after = pd.to_datetime(data[col], errors='coerce').notna().sum()
                if non_null_after > len(data) * 0.5:  # More than 50% can be converted
                    suggestions.append({
                        'Column': col,
                        'Current Type': 'object',
                        'Suggested Type': 'datetime',
                        'Reason': f'Can convert {non_null_after} values to datetime'
                    })
            except:
                pass
    
    if suggestions:
        st.dataframe(pd.DataFrame(suggestions), use_container_width=True)
    else:
        st.info("No type conversion suggestions found.")

elif selected_issue == 'outliers':
    st.subheader("Outlier Detection")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
        
        col_data = data[selected_col].dropna()
        
        # Calculate outliers using IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Outliers", len(outliers))
            st.metric("Outlier Percentage", f"{len(outliers)/len(col_data)*100:.2f}%")
            st.metric("Lower Bound", f"{lower_bound:.2f}")
            st.metric("Upper Bound", f"{upper_bound:.2f}")
        
        with col2:
            # Box plot
            fig = px.box(y=col_data, title=f'Box Plot - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Histogram
        fig = px.histogram(col_data, title=f'Distribution - {selected_col}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Z-score outliers
        z_scores = np.abs(stats.zscore(col_data))
        z_outliers = col_data[z_scores > 3]
        
        st.subheader("Z-Score Outliers (|z| > 3)")
        st.metric("Z-Score Outliers", len(z_outliers))
        
        if len(z_outliers) > 0:
            st.dataframe(pd.DataFrame({'Values': z_outliers, 'Z-Score': z_scores[z_scores > 3]}))
    
    else:
        st.info("No numeric columns found for outlier analysis.")

elif selected_issue == 'duplicate_rows':
    st.subheader("Duplicate Rows Analysis")
    
    duplicates = data.duplicated()
    duplicate_count = duplicates.sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Duplicates", duplicate_count)
        st.metric("Duplicate Percentage", f"{duplicate_count/len(data)*100:.2f}%")
    
    with col2:
        # Duplicate distribution
        fig = px.pie(
            values=[len(data) - duplicate_count, duplicate_count],
            names=['Unique', 'Duplicates'],
            title='Duplicate vs Unique Rows'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if duplicate_count > 0:
        st.subheader("Duplicate Rows Preview")
        duplicate_rows = data[duplicates]
        st.dataframe(duplicate_rows.head(10), use_container_width=True)
        
        # Column-wise duplicate analysis
        st.subheader("Column-wise Duplicate Analysis")
        col_duplicates = []
        for col in data.columns:
            col_dup_count = data[col].duplicated().sum()
            col_duplicates.append({
                'Column': col,
                'Duplicate Count': col_dup_count,
                'Unique Values': data[col].nunique(),
                'Duplicate Percentage': f"{col_dup_count/len(data)*100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(col_duplicates), use_container_width=True)
    
    else:
        st.success("🎉 No duplicate rows found!")

elif selected_issue == 'inconsistent_categories':
    st.subheader("Inconsistent Categories Analysis")
    
    cat_cols = data.select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        selected_cat_col = st.selectbox("Select categorical column:", cat_cols)
        
        col_values = data[selected_cat_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Value Counts**")
            st.dataframe(col_values.head(20), use_container_width=True)
        
        with col2:
            # Value distribution
            fig = px.bar(
                x=col_values.head(10).index,
                y=col_values.head(10).values,
                title=f'Top 10 Values in {selected_cat_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Potential inconsistencies
        st.subheader("Potential Inconsistencies")
        unique_values = data[selected_cat_col].dropna().unique()
        
        # Check for case variations
        case_issues = []
        for val in unique_values:
            if isinstance(val, str):
                similar_vals = [v for v in unique_values if isinstance(v, str) and v.lower() == val.lower() and v != val]
                if similar_vals:
                    case_issues.append({
                        'Value': val,
                        'Similar Values': similar_vals,
                        'Issue Type': 'Case Variation'
                    })
        
        if case_issues:
            st.dataframe(pd.DataFrame(case_issues), use_container_width=True)
        else:
            st.info("No obvious case inconsistencies found.")
    
    else:
        st.info("No categorical columns found.")

elif selected_issue == 'class_imbalance':
    st.subheader("Class Imbalance Analysis")
    
    cat_cols = data.select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        selected_target = st.selectbox("Select target variable:", cat_cols)
        
        value_counts = data[selected_target].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Class Distribution**")
            class_df = pd.DataFrame({
                'Class': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(data) * 100).round(2)
            })
            st.dataframe(class_df, use_container_width=True)
        
        with col2:
            # Class distribution pie chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Class Distribution - {selected_target}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Imbalance metrics
        st.subheader("Imbalance Metrics")
        majority_class = value_counts.max()
        minority_class = value_counts.min()
        imbalance_ratio = majority_class / minority_class
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Majority Class Count", majority_class)
        
        with metric_col2:
            st.metric("Minority Class Count", minority_class)
        
        with metric_col3:
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            st.warning("⚠️ Significant class imbalance detected!")
        else:
            st.success("✅ Classes are reasonably balanced.")
    
    else:
        st.info("No categorical columns found for class imbalance analysis.")

elif selected_issue == 'skewed_distributions':
    st.subheader("Skewed Distributions Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Calculate skewness for all numeric columns
        skewness_data = []
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                skew_val = stats.skew(col_data)
                skewness_data.append({
                    'Column': col,
                    'Skewness': skew_val,
                    'Interpretation': 'Highly Skewed' if abs(skew_val) > 1 else 'Moderately Skewed' if abs(skew_val) > 0.5 else 'Approximately Normal'
                })
        
        skew_df = pd.DataFrame(skewness_data)
        st.dataframe(skew_df, use_container_width=True)
        
        # Visualize skewed columns
        highly_skewed = skew_df[skew_df['Skewness'].abs() > 1]['Column'].tolist()
        
        if highly_skewed:
            st.subheader("Highly Skewed Columns Visualization")
            selected_skew_col = st.selectbox("Select column to visualize:", highly_skewed)
            
            col_data = data[selected_skew_col].dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Original distribution
                fig = px.histogram(col_data, title=f'Original Distribution - {selected_skew_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Log transformation
                if col_data.min() > 0:
                    log_data = np.log(col_data)
                    fig = px.histogram(log_data, title=f'Log-Transformed Distribution - {selected_skew_col}')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Cannot apply log transformation (contains zero or negative values)")
        
        else:
            st.success("✅ No highly skewed columns found!")
    
    else:
        st.info("No numeric columns found for skewness analysis.")

elif selected_issue == 'low_variance_columns':
    st.subheader("Low Variance Columns Analysis")
    
    # Analyze all columns for low variance
    low_variance_cols = []
    
    for col in data.columns:
        if data[col].dtype in ['object']:
            # For categorical columns, check unique values
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.01:  # Less than 1% unique values
                low_variance_cols.append({
                    'Column': col,
                    'Type': 'Categorical',
                    'Unique Values': data[col].nunique(),
                    'Unique Ratio': f"{unique_ratio*100:.2f}%",
                    'Most Common': data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A'
                })
        else:
            # For numeric columns, check variance
            if data[col].var() < 0.01:  # Very low variance
                low_variance_cols.append({
                    'Column': col,
                    'Type': 'Numeric',
                    'Variance': data[col].var(),
                    'Std Dev': data[col].std(),
                    'Range': data[col].max() - data[col].min()
                })
    
    if low_variance_cols:
        st.dataframe(pd.DataFrame(low_variance_cols), use_container_width=True)
        st.warning("⚠️ These columns have very low variance and might not be useful for analysis.")
    else:
        st.success("✅ No low variance columns detected!")

elif selected_issue == 'multicollinearity':
    st.subheader("Multicollinearity Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Find highly correlated pairs
        st.subheader("Highly Correlated Pairs")
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'Column 1': corr_matrix.columns[i],
                        'Column 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                        'Strength': 'Very High' if abs(corr_val) > 0.9 else 'High'
                    })
        
        if high_corr_pairs:
            st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
            st.warning("⚠️ High correlation detected! Consider removing one variable from each pair.")
        else:
            st.success("✅ No high multicollinearity detected!")
    
    else:
        st.info("Need at least 2 numeric columns for multicollinearity analysis.")

elif selected_issue == 'invalid_values':
    st.subheader("Invalid/Impossible Values Analysis")
    
    invalid_issues = []
    
    # Check each column for domain-specific invalid values
    for col in data.columns:
        col_data = data[col].dropna()
        
        # Age columns (should be positive and reasonable)
        if 'age' in col.lower():
            invalid_ages = col_data[(col_data < 0) | (col_data > 150)]
            if len(invalid_ages) > 0:
                invalid_issues.append({
                    'Column': col,
                    'Issue': 'Invalid Age',
                    'Count': len(invalid_ages),
                    'Examples': invalid_ages.head(5).tolist()
                })
        
        # Percentage columns (should be 0-100)
        if 'percent' in col.lower() or 'rate' in col.lower():
            invalid_pct = col_data[(col_data < 0) | (col_data > 100)]
            if len(invalid_pct) > 0:
                invalid_issues.append({
                    'Column': col,
                    'Issue': 'Invalid Percentage',
                    'Count': len(invalid_pct),
                    'Examples': invalid_pct.head(5).tolist()
                })
        
        # Salary/Price columns (should be positive)
        if any(keyword in col.lower() for keyword in ['salary', 'price', 'cost', 'amount']):
            invalid_salary = col_data[col_data < 0]
            if len(invalid_salary) > 0:
                invalid_issues.append({
                    'Column': col,
                    'Issue': 'Negative Value',
                    'Count': len(invalid_salary),
                    'Examples': invalid_salary.head(5).tolist()
                })
    
    if invalid_issues:
        st.dataframe(pd.DataFrame(invalid_issues), use_container_width=True)
        st.warning("⚠️ Invalid values detected based on domain knowledge!")
    else:
        st.success("✅ No obvious invalid values detected!")

# Issue summary
st.header("📊 Issues Summary")

summary_data = []
for issue_key, issue_name in issue_options.items():
    issue_data = data_issues.get(issue_key, {})
    summary_data.append({
        'Issue': issue_name,
        'Severity': issue_data.get('severity', 'Unknown'),
        'Count': issue_data.get('count', 0),
        'Status': '🔴 Needs Attention' if issue_data.get('count', 0) > 0 else '✅ OK'
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Recommendations
st.header("💡 Recommendations")

recommendations = [
    "🧹 **Data Cleaning**: Address missing values and outliers before analysis",
    "🔄 **Data Transformation**: Apply appropriate transformations for skewed distributions",
    "📊 **Feature Engineering**: Consider creating new features from existing ones",
    "⚖️ **Balancing**: Use sampling techniques for imbalanced datasets",
    "🔍 **Validation**: Implement data validation rules to prevent future issues"
]

for rec in recommendations:
    st.markdown(rec)

st.markdown("---")
st.markdown("**Data Quality Analysis** - Comprehensive assessment of common data issues")

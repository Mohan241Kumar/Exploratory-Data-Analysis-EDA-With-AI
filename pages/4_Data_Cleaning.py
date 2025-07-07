import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Cleaning & Preprocessing")
st.markdown("Apply cleaning operations and download cleaned datasets")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload a dataset first from the main page.")
    st.stop()

# Initialize cleaned data
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

original_data = st.session_state.original_data
cleaned_data = st.session_state.cleaned_data

# Data cleaning options
st.header("🛠️ Cleaning Operations")

# Sidebar for cleaning options
st.sidebar.header("🧹 Cleaning Options")

cleaning_options = {
    "Handle Missing Data": st.sidebar.checkbox("Handle Missing Data", value=True),
    "Remove Duplicates": st.sidebar.checkbox("Remove Duplicates", value=True),
    "Fix Data Types": st.sidebar.checkbox("Fix Data Types", value=False),
    "Handle Outliers": st.sidebar.checkbox("Handle Outliers", value=False),
    "Normalize/Scale Data": st.sidebar.checkbox("Normalize/Scale Data", value=False),
    "Fix Inconsistent Categories": st.sidebar.checkbox("Fix Inconsistent Categories", value=False),
    "Remove Low Variance Columns": st.sidebar.checkbox("Remove Low Variance Columns", value=False)
}

# Apply cleaning button
if st.sidebar.button("🚀 Apply All Selected Cleaning Operations"):
    cleaned_data = original_data.copy()
    cleaning_log = []
    
    # 1. Handle Missing Data
    if cleaning_options["Handle Missing Data"]:
        st.subheader("🔧 Handling Missing Data")
        
        missing_before = cleaned_data.isnull().sum().sum()
        
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().sum() > 0:
                if cleaned_data[col].dtype in ['int64', 'float64']:
                    # Numeric: impute with median
                    median_val = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_val, inplace=True)
                    cleaning_log.append(f"Filled missing values in '{col}' with median: {median_val}")
                else:
                    # Categorical: impute with mode
                    mode_val = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown'
                    cleaned_data[col].fillna(mode_val, inplace=True)
                    cleaning_log.append(f"Filled missing values in '{col}' with mode: '{mode_val}'")
        
        missing_after = cleaned_data.isnull().sum().sum()
        cleaning_log.append(f"Missing values reduced from {missing_before} to {missing_after}")
    
    # 2. Remove Duplicates
    if cleaning_options["Remove Duplicates"]:
        duplicates_before = cleaned_data.duplicated().sum()
        cleaned_data.drop_duplicates(inplace=True)
        duplicates_after = cleaned_data.duplicated().sum()
        cleaning_log.append(f"Removed {duplicates_before} duplicate rows")
    
    # 3. Fix Data Types
    if cleaning_options["Fix Data Types"]:
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(cleaned_data[col], errors='coerce')
                    if numeric_series.notna().sum() > len(cleaned_data) * 0.8:  # 80% can be converted
                        cleaned_data[col] = numeric_series
                        cleaning_log.append(f"Converted '{col}' to numeric type")
                except:
                    pass
                
                # Try to convert to datetime
                try:
                    datetime_series = pd.to_datetime(cleaned_data[col], errors='coerce')
                    if datetime_series.notna().sum() > len(cleaned_data) * 0.8:  # 80% can be converted
                        cleaned_data[col] = datetime_series
                        cleaning_log.append(f"Converted '{col}' to datetime type")
                except:
                    pass
    
    # 4. Handle Outliers
    if cleaning_options["Handle Outliers"]:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
            
            if outliers_before > 0:
                # Cap outliers instead of removing them
                cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                cleaning_log.append(f"Capped {outliers_before} outliers in '{col}'")
    
    # 5. Normalize/Scale Data
    if cleaning_options["Normalize/Scale Data"]:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            cleaned_data[numeric_cols] = scaler.fit_transform(cleaned_data[numeric_cols])
            cleaning_log.append(f"Standardized {len(numeric_cols)} numeric columns")
    
    # 6. Fix Inconsistent Categories
    if cleaning_options["Fix Inconsistent Categories"]:
        cat_cols = cleaned_data.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            # Convert to lowercase and strip whitespace
            original_unique = cleaned_data[col].nunique()
            cleaned_data[col] = cleaned_data[col].astype(str).str.lower().str.strip()
            new_unique = cleaned_data[col].nunique()
            
            if original_unique != new_unique:
                cleaning_log.append(f"Fixed inconsistent categories in '{col}': {original_unique} → {new_unique} unique values")
    
    # 7. Remove Low Variance Columns
    if cleaning_options["Remove Low Variance Columns"]:
        low_var_cols = []
        
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype in ['int64', 'float64']:
                if cleaned_data[col].var() < 0.01:  # Very low variance
                    low_var_cols.append(col)
            else:
                if cleaned_data[col].nunique() == 1:  # Only one unique value
                    low_var_cols.append(col)
        
        if low_var_cols:
            cleaned_data.drop(columns=low_var_cols, inplace=True)
            cleaning_log.append(f"Removed {len(low_var_cols)} low variance columns: {low_var_cols}")
    
    # Update session state
    st.session_state.cleaned_data = cleaned_data
    st.session_state.cleaning_log = cleaning_log
    
    st.success(f"✅ Cleaning completed! Applied {sum(cleaning_options.values())} operations.")

# Display cleaning log
if 'cleaning_log' in st.session_state:
    st.header("📋 Cleaning Log")
    
    for i, log_entry in enumerate(st.session_state.cleaning_log, 1):
        st.write(f"{i}. {log_entry}")

# Before/After Comparison
st.header("⚖️ Before vs After Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Original Data")
    st.write(f"**Shape:** {original_data.shape[0]:,} rows × {original_data.shape[1]:,} columns")
    st.write(f"**Missing Values:** {original_data.isnull().sum().sum():,}")
    st.write(f"**Duplicates:** {original_data.duplicated().sum():,}")
    st.write(f"**Memory Usage:** {original_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

with col2:
    st.subheader("🧹 Cleaned Data")
    st.write(f"**Shape:** {cleaned_data.shape[0]:,} rows × {cleaned_data.shape[1]:,} columns")
    st.write(f"**Missing Values:** {cleaned_data.isnull().sum().sum():,}")
    st.write(f"**Duplicates:** {cleaned_data.duplicated().sum():,}")
    st.write(f"**Memory Usage:** {cleaned_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data type comparison
st.subheader("📊 Data Type Comparison")

dtype_comparison = pd.DataFrame({
    'Column': original_data.columns,
    'Original Type': original_data.dtypes.astype(str),
    'Cleaned Type': cleaned_data.dtypes.astype(str) if cleaned_data.shape[1] > 0 else ['N/A'] * len(original_data.columns),
    'Changed': ['✅' if orig != clean else '➖' for orig, clean in zip(original_data.dtypes.astype(str), cleaned_data.dtypes.astype(str) if cleaned_data.shape[1] > 0 else ['N/A'] * len(original_data.columns))]
})

st.dataframe(dtype_comparison, use_container_width=True)

# Data preview comparison
st.subheader("👀 Data Preview Comparison")

preview_tabs = st.tabs(["Original Data", "Cleaned Data", "Side-by-Side"])

with preview_tabs[0]:
    st.dataframe(original_data.head(10), use_container_width=True)

with preview_tabs[1]:
    st.dataframe(cleaned_data.head(10), use_container_width=True)

with preview_tabs[2]:
    if cleaned_data.shape[1] > 0:
        comparison_cols = st.columns(2)
        
        with comparison_cols[0]:
            st.write("**Original Data (First 5 rows)**")
            st.dataframe(original_data.head(5), use_container_width=True)
        
        with comparison_cols[1]:
            st.write("**Cleaned Data (First 5 rows)**")
            st.dataframe(cleaned_data.head(5), use_container_width=True)

# Manual cleaning operations
st.header("🛠️ Manual Cleaning Operations")

manual_tabs = st.tabs(["Column Operations", "Row Operations", "Value Transformations"])

with manual_tabs[0]:
    st.subheader("Column Operations")
    
    # Drop columns
    cols_to_drop = st.multiselect("Select columns to drop:", cleaned_data.columns.tolist())
    
    if st.button("Drop Selected Columns") and cols_to_drop:
        cleaned_data = cleaned_data.drop(columns=cols_to_drop)
        st.session_state.cleaned_data = cleaned_data
        st.success(f"Dropped {len(cols_to_drop)} columns")
        st.rerun()
    
    # Rename columns
    st.subheader("Rename Columns")
    
    col_to_rename = st.selectbox("Select column to rename:", cleaned_data.columns.tolist())
    new_name = st.text_input("New column name:")
    
    if st.button("Rename Column") and new_name:
        cleaned_data = cleaned_data.rename(columns={col_to_rename: new_name})
        st.session_state.cleaned_data = cleaned_data
        st.success(f"Renamed '{col_to_rename}' to '{new_name}'")
        st.rerun()

with manual_tabs[1]:
    st.subheader("Row Operations")
    
    # Filter rows
    st.write("**Filter Rows**")
    
    filter_col = st.selectbox("Select column for filtering:", cleaned_data.columns.tolist())
    
    if cleaned_data[filter_col].dtype == 'object':
        # Categorical filter
        unique_vals = cleaned_data[filter_col].unique()
        selected_vals = st.multiselect("Select values to keep:", unique_vals, default=unique_vals.tolist())
        
        if st.button("Apply Row Filter"):
            cleaned_data = cleaned_data[cleaned_data[filter_col].isin(selected_vals)]
            st.session_state.cleaned_data = cleaned_data
            st.success(f"Filtered to {len(cleaned_data)} rows")
            st.rerun()
    
    else:
        # Numeric filter
        min_val = float(cleaned_data[filter_col].min())
        max_val = float(cleaned_data[filter_col].max())
        filter_range = st.slider("Select range to keep:", min_val, max_val, (min_val, max_val))
        
        if st.button("Apply Range Filter"):
            cleaned_data = cleaned_data[
                (cleaned_data[filter_col] >= filter_range[0]) & 
                (cleaned_data[filter_col] <= filter_range[1])
            ]
            st.session_state.cleaned_data = cleaned_data
            st.success(f"Filtered to {len(cleaned_data)} rows")
            st.rerun()

with manual_tabs[2]:
    st.subheader("Value Transformations")
    
    # Transform column values
    transform_col = st.selectbox("Select column to transform:", cleaned_data.columns.tolist())
    
    if cleaned_data[transform_col].dtype in ['int64', 'float64']:
        # Numeric transformations
        transform_type = st.selectbox("Select transformation:", [
            "None", "Log Transform", "Square Root", "Square", "Normalize (0-1)", "Standardize (z-score)"
        ])
        
        if st.button("Apply Transformation") and transform_type != "None":
            if transform_type == "Log Transform":
                if cleaned_data[transform_col].min() > 0:
                    cleaned_data[transform_col] = np.log(cleaned_data[transform_col])
                    st.success("Applied log transformation")
                else:
                    st.error("Cannot apply log transformation to zero or negative values")
            
            elif transform_type == "Square Root":
                if cleaned_data[transform_col].min() >= 0:
                    cleaned_data[transform_col] = np.sqrt(cleaned_data[transform_col])
                    st.success("Applied square root transformation")
                else:
                    st.error("Cannot apply square root transformation to negative values")
            
            elif transform_type == "Square":
                cleaned_data[transform_col] = cleaned_data[transform_col] ** 2
                st.success("Applied square transformation")
            
            elif transform_type == "Normalize (0-1)":
                min_val = cleaned_data[transform_col].min()
                max_val = cleaned_data[transform_col].max()
                cleaned_data[transform_col] = (cleaned_data[transform_col] - min_val) / (max_val - min_val)
                st.success("Applied normalization")
            
            elif transform_type == "Standardize (z-score)":
                mean_val = cleaned_data[transform_col].mean()
                std_val = cleaned_data[transform_col].std()
                cleaned_data[transform_col] = (cleaned_data[transform_col] - mean_val) / std_val
                st.success("Applied standardization")
            
            st.session_state.cleaned_data = cleaned_data
            st.rerun()
    
    else:
        # String transformations
        transform_type = st.selectbox("Select transformation:", [
            "None", "Lowercase", "Uppercase", "Title Case", "Strip Whitespace"
        ])
        
        if st.button("Apply String Transformation") and transform_type != "None":
            if transform_type == "Lowercase":
                cleaned_data[transform_col] = cleaned_data[transform_col].str.lower()
                st.success("Applied lowercase transformation")
            
            elif transform_type == "Uppercase":
                cleaned_data[transform_col] = cleaned_data[transform_col].str.upper()
                st.success("Applied uppercase transformation")
            
            elif transform_type == "Title Case":
                cleaned_data[transform_col] = cleaned_data[transform_col].str.title()
                st.success("Applied title case transformation")
            
            elif transform_type == "Strip Whitespace":
                cleaned_data[transform_col] = cleaned_data[transform_col].str.strip()
                st.success("Applied whitespace stripping")
            
            st.session_state.cleaned_data = cleaned_data
            st.rerun()

# Quality metrics
st.header("📊 Data Quality Metrics")

quality_metrics = pd.DataFrame({
    'Metric': [
        'Completeness',
        'Uniqueness',
        'Consistency',
        'Validity'
    ],
    'Original Data': [
        f"{(1 - original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1])) * 100:.1f}%",
        f"{(1 - original_data.duplicated().sum() / len(original_data)) * 100:.1f}%",
        "Manual Review Required",
        "Manual Review Required"
    ],
    'Cleaned Data': [
        f"{(1 - cleaned_data.isnull().sum().sum() / (cleaned_data.shape[0] * cleaned_data.shape[1])) * 100:.1f}%",
        f"{(1 - cleaned_data.duplicated().sum() / len(cleaned_data)) * 100:.1f}%",
        "Manual Review Required",
        "Manual Review Required"
    ]
})

st.dataframe(quality_metrics, use_container_width=True)

# Export options
st.header("📥 Export Options")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    st.subheader("📊 Export Cleaned Data")
    
    if st.button("📥 Download Cleaned CSV"):
        csv_data = cleaned_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

with export_col2:
    st.subheader("📋 Export Cleaning Report")
    
    if st.button("📥 Download Cleaning Report"):
        report_data = {
            'Original Data Shape': original_data.shape,
            'Cleaned Data Shape': cleaned_data.shape,
            'Cleaning Operations': st.session_state.get('cleaning_log', []),
            'Data Quality Metrics': quality_metrics.to_dict('records')
        }
        
        import json
        report_json = json.dumps(report_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Cleaning Report",
            data=report_json,
            file_name="cleaning_report.json",
            mime="application/json"
        )

with export_col3:
    st.subheader("🔄 Reset Data")
    
    if st.button("🔄 Reset to Original"):
        st.session_state.cleaned_data = st.session_state.original_data.copy()
        if 'cleaning_log' in st.session_state:
            del st.session_state.cleaning_log
        st.success("Data reset to original state")
        st.rerun()

st.markdown("---")
st.markdown("**Data Cleaning** - Transform and prepare your data for analysis")

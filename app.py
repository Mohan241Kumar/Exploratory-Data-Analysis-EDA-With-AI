import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.eda_utils import load_data, get_basic_info
from utils.data_quality import detect_all_issues
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Advanced EDA Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'data_issues' not in st.session_state:
    st.session_state.data_issues = None

# Main title
st.title("🔍 Advanced EDA Tool")
st.markdown("**Comprehensive Exploratory Data Analysis with Automated Profiling & Data Quality Assessment**")

# Sidebar for file upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload your CSV file for comprehensive EDA analysis"
)

if uploaded_file is not None:
    try:
        # Load data
        data = load_data(uploaded_file)
        st.session_state.data = data
        st.session_state.original_data = data.copy()
        
        # Detect data quality issues
        st.session_state.data_issues = detect_all_issues(data)
        
        st.sidebar.success(f"✅ File uploaded successfully!")
        st.sidebar.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.stop()

# Main content
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Overview section
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")
    
    with col2:
        st.metric("Columns", f"{data.shape[1]:,}")
    
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    
    with col4:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data quality issues summary
    if st.session_state.data_issues:
        st.header("🚨 Data Quality Issues Summary")
        
        issues_summary = []
        for issue_type, details in st.session_state.data_issues.items():
            if details['count'] > 0:
                issues_summary.append({
                    'Issue Type': issue_type.replace('_', ' ').title(),
                    'Count': details['count'],
                    'Severity': details.get('severity', 'Medium'),
                    'Description': details.get('description', '')
                })
        
        if issues_summary:
            issues_df = pd.DataFrame(issues_summary)
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("🎉 No major data quality issues detected!")
    
    # Quick data preview
    st.header("👀 Data Preview")
    
    preview_tabs = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample", "Data Types"])
    
    with preview_tabs[0]:
        st.dataframe(data.head(10), use_container_width=True)
    
    with preview_tabs[1]:
        st.dataframe(data.tail(10), use_container_width=True)
    
    with preview_tabs[2]:
        st.dataframe(data.sample(min(10, len(data))), use_container_width=True)
    
    with preview_tabs[3]:
        dtypes_df = pd.DataFrame({
            'Column': data.dtypes.index,
            'Data Type': data.dtypes.values.astype(str),
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values
        })
        st.dataframe(dtypes_df, use_container_width=True)
    
    # Basic statistics
    st.header("📈 Basic Statistics")
    
    stat_type = st.selectbox(
        "Select Statistics Type",
        ["Numerical", "Categorical", "All"]
    )
    
    if stat_type == "Numerical":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(data[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
    
    elif stat_type == "Categorical":
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_stats = []
            for col in cat_cols:
                cat_stats.append({
                    'Column': col,
                    'Unique Values': data[col].nunique(),
                    'Most Frequent': data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A',
                    'Frequency': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
                })
            st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset.")
    
    else:
        st.dataframe(data.describe(include='all'), use_container_width=True)
    
    # Navigation info
    st.header("🧭 Navigation Guide")
    st.markdown("""
    Use the sidebar to navigate between different analysis sections:
    
    - **🤖 Automated Profiling**: Generate comprehensive reports using Pandas-Profiling, Sweetviz, and Autoviz
    - **🔍 Data Quality Issues**: Detailed analysis of all 10 common data quality problems
    - **📊 Interactive Analysis**: Custom visualizations and exploratory analysis
    - **🧹 Data Cleaning**: Apply cleaning operations and download cleaned datasets
    - **🤖 AI Data Cleaning**: Intelligent AI-powered data cleaning with smart recommendations
    - **📊 Auto Dashboard**: AI-generated dashboards similar to Power BI with automated insights
    - **💬 Question Answering**: Ask questions about your data using text
    - **🧠 ML Recommendations**: Get intelligent machine learning model recommendations
    - **📊 3D Visualizations**: Explore your data in three dimensions with various 3D plot types

    """)

else:
    # Welcome screen
    st.header("Welcome to the Advanced EDA Tool!")
    
    st.markdown("""
    This comprehensive EDA tool helps you analyze your data and identify quality issues automatically.
    
    ### 🌟 Key Features:
    - **Automated Profiling**: Generate detailed reports using industry-standard libraries
    - **Quality Assessment**: Detect and visualize 10 common data quality issues
    - **Interactive Analysis**: Explore your data with dynamic visualizations
    - **Data Cleaning**: Apply transformations and export cleaned datasets

    
    ### 📁 Getting Started:
    1. Upload a CSV file using the sidebar
    2. Explore the automated analysis results
    3. Navigate through different analysis sections
    4. Download cleaned data and reports
    
    ### 🔧 Supported Issues Detection:
    1. Missing Data
    2. Data Type Mismatches
    3. Outliers
    4. Duplicate Rows
    5. Inconsistent Categories
    6. Unbalanced Classes
    7. Skewed Distributions
    8. Low Variance Columns
    9. Multicollinearity
    10. Invalid/Impossible Values
    
    ### 🤖 AI-Powered Features:
    - **Smart Data Cleaning**: AI recommendations for optimal cleaning strategies
    - **Automated Dashboards**: Power BI-style dashboards with AI-generated insights
    - **Question Answering**: Natural language Q&A (text only)
    - **Intelligent Analysis**: Natural language explanations of data patterns
    - **Business Insights**: AI-driven recommendations for data-driven decisions

    """)
    
    # Sample data section
    st.header("📊 Try with Sample Data")
    
    if st.button("Generate Sample Dataset"):
        np.random.seed(42)
        
        # Create sample data with various quality issues
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'id': range(1, n_samples + 1),
            'age': np.random.normal(35, 10, n_samples),
            'salary': np.random.exponential(50000, n_samples),
            'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'hr', 'it'], n_samples),
            'experience': np.random.normal(8, 4, n_samples),
            'rating': np.random.uniform(1, 5, n_samples),
            'constant_col': 'A',
            'mostly_null': [np.nan] * 950 + ['value'] * 50,
            'duplicate_col': np.random.choice(['X', 'Y'], n_samples)
        })
        
        # Introduce some quality issues
        sample_data.loc[np.random.choice(sample_data.index, 50), 'age'] = np.nan
        sample_data.loc[np.random.choice(sample_data.index, 20), 'age'] = -5  # Invalid age
        sample_data.loc[np.random.choice(sample_data.index, 30), 'salary'] = sample_data['salary'].mean() * 10  # Outliers
        
        # Add some duplicates
        sample_data = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
        
        st.session_state.data = sample_data
        st.session_state.original_data = sample_data.copy()
        st.session_state.data_issues = detect_all_issues(sample_data)
        
        st.success("Sample dataset generated! Refresh the page to see the analysis.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Advanced EDA Tool** - Comprehensive Data Analysis and Quality Assessment")

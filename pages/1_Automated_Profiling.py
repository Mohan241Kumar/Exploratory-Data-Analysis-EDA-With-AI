import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
import tempfile
import os
import base64
from autoviz.AutoViz_Class import AutoViz_Class
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Automated Profiling",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Automated Profiling")
st.markdown("Generate comprehensive EDA reports using industry-standard libraries")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

# Profiling options
st.header("📊 Profiling Options")

col1, col2, col3 = st.columns(3)

with col1:
    generate_pandas_profiling = st.button("🐼 Generate Pandas Profiling Report", use_container_width=True)

with col2:
    generate_sweetviz = st.button("🍯 Generate Sweetviz Report", use_container_width=True)

with col3:
    generate_autoviz = st.button("🎨 Generate AutoViz Report", use_container_width=True)

# Pandas Profiling
if generate_pandas_profiling:
    st.header("🐼 Pandas Profiling Report")
    
    with st.spinner("Generating comprehensive profiling report..."):
        try:
            # Create profile report
            profile = ProfileReport(
                data,
                title="Pandas Profiling Report",
                explorative=True,
                minimal=False
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                profile.to_file(tmp_file.name)
                
                # Read the HTML content
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Display in iframe
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Pandas Profiling Report",
                    data=html_content,
                    file_name="pandas_profiling_report.html",
                    mime="text/html"
                )
                
                # Clean up
                os.unlink(tmp_file.name)
                
        except Exception as e:
            st.error(f"Error generating Pandas Profiling report: {str(e)}")

# Sweetviz
if generate_sweetviz:
    st.header("🍯 Sweetviz Report")
    
    with st.spinner("Generating Sweetviz analysis..."):
        try:
            # Create Sweetviz report
            report = sv.analyze(data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                report.show_html(tmp_file.name, open_browser=False)
                
                # Read the HTML content
                with open(tmp_file.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Display in iframe
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Sweetviz Report",
                    data=html_content,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )
                
                # Clean up
                os.unlink(tmp_file.name)
                
        except Exception as e:
            st.error(f"Error generating Sweetviz report: {str(e)}")

# AutoViz
if generate_autoviz:
    st.header("🎨 AutoViz Report")
    
    with st.spinner("Generating AutoViz visualizations..."):
        try:
            # Create AutoViz instance
            av = AutoViz_Class()
            
            # Generate plots
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save data to temporary CSV
                temp_csv = os.path.join(temp_dir, "temp_data.csv")
                data.to_csv(temp_csv, index=False)
                
                # Generate AutoViz report
                av.AutoViz(
                    filename=temp_csv,
                    sep=',',
                    depVar='',
                    dfte=None,
                    header=0,
                    verbose=0,
                    lowess=False,
                    chart_format='png',
                    max_rows_analyzed=2000,
                    max_cols_analyzed=30,
                    save_plot_dir=temp_dir
                )
                
                # Display generated plots
                plot_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
                
                if plot_files:
                    st.success(f"Generated {len(plot_files)} visualizations")
                    
                    # Display plots in columns
                    cols = st.columns(2)
                    for i, plot_file in enumerate(plot_files):
                        with cols[i % 2]:
                            plot_path = os.path.join(temp_dir, plot_file)
                            if os.path.exists(plot_path):
                                st.image(plot_path, caption=plot_file.replace('.png', ''))
                else:
                    st.info("No visualizations were generated by AutoViz.")
                    
        except Exception as e:
            st.error(f"Error generating AutoViz report: {str(e)}")

# Manual profiling summary
st.header("📋 Quick Profiling Summary")

# Dataset overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    overview_data = {
        'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicate Rows', 'Memory Usage (MB)'],
        'Value': [
            f"{data.shape[0]:,}",
            f"{data.shape[1]:,}",
            f"{data.isnull().sum().sum():,}",
            f"{data.duplicated().sum():,}",
            f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

with col2:
    st.subheader("Column Types")
    dtype_counts = data.dtypes.value_counts()
    dtype_data = {
        'Data Type': dtype_counts.index.astype(str),
        'Count': dtype_counts.values
    }
    st.dataframe(pd.DataFrame(dtype_data), use_container_width=True)

# Missing data analysis
st.subheader("Missing Data Analysis")
missing_data = data.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': (missing_data.values / len(data) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)
else:
    st.success("No missing values found!")

# Numerical columns summary
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    st.subheader("Numerical Columns Summary")
    st.dataframe(data[numeric_cols].describe(), use_container_width=True)

# Categorical columns summary
cat_cols = data.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    st.subheader("Categorical Columns Summary")
    cat_summary = []
    for col in cat_cols:
        cat_summary.append({
            'Column': col,
            'Unique Values': data[col].nunique(),
            'Most Frequent': data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A',
            'Frequency': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
        })
    st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

# Export options
st.header("📥 Export Options")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📊 Export Dataset Summary"):
        summary_data = {
            'Dataset Overview': overview_data,
            'Missing Data': missing_df.to_dict('records') if len(missing_data) > 0 else [],
            'Numerical Summary': data[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            'Categorical Summary': cat_summary if len(cat_cols) > 0 else []
        }
        
        st.download_button(
            label="📥 Download Summary JSON",
            data=pd.Series(summary_data).to_json(indent=2),
            file_name="dataset_summary.json",
            mime="application/json"
        )

with export_col2:
    if st.button("📈 Export Basic Statistics"):
        stats_data = data.describe(include='all')
        csv_data = stats_data.to_csv()
        
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv_data,
            file_name="dataset_statistics.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("**Automated Profiling** - Generate comprehensive reports with industry-standard tools")

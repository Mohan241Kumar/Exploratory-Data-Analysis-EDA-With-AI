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

# Fix numpy compatibility issue with Sweetviz
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = UserWarning

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
            # Handle numpy compatibility issue
            if not hasattr(np, 'VisibleDeprecationWarning'):
                np.VisibleDeprecationWarning = UserWarning
            
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
            st.info("💡 This is likely due to numpy compatibility with Sweetviz. Using alternative summary instead.")
            
            # Provide alternative summary
            st.subheader("📊 Alternative Data Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Overview:**")
                st.write(f"- Rows: {data.shape[0]:,}")
                st.write(f"- Columns: {data.shape[1]:,}")
                st.write(f"- Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
            with col2:
                st.write("**Data Quality:**")
                missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                st.write(f"- Missing values: {missing_pct:.1f}%")
                st.write(f"- Duplicate rows: {data.duplicated().sum():,}")
                st.write(f"- Numeric columns: {len(data.select_dtypes(include=[np.number]).columns)}")
            
            # Show data types distribution
            import plotly.express as px
            st.write("**Column Types:**")
            type_counts = data.dtypes.value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, title="Data Types Distribution")
            st.plotly_chart(fig, use_container_width=True)

# AutoViz
if generate_autoviz:
    st.header("🎨 AutoViz Report")
    
    with st.spinner("Generating AutoViz visualizations..."):
        try:
            # Create AutoViz instance
            av = AutoViz_Class()
            
            # Generate plots using AutoViz
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create a temporary directory for plots
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate AutoViz report with data directly
                dfte = av.AutoViz(
                    filename='',
                    sep=',',
                    depVar='',
                    dfte=data,
                    header=0,
                    verbose=0,
                    lowess=False,
                    chart_format='png',
                    max_rows_analyzed=2000,
                    max_cols_analyzed=30,
                    save_plot_dir=temp_dir
                )
                
                # Look for generated plots
                plot_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.png') or file.endswith('.svg'):
                            plot_files.append(os.path.join(root, file))
                
                if plot_files:
                    st.success(f"Generated {len(plot_files)} visualizations")
                    
                    # Display plots
                    for i, plot_path in enumerate(plot_files):
                        if os.path.exists(plot_path):
                            plot_name = os.path.basename(plot_path).replace('.png', '').replace('.svg', '')
                            st.subheader(f"📊 {plot_name}")
                            st.image(plot_path, use_column_width=True)
                else:
                    # If no plots generated, create alternative visualizations
                    st.info("AutoViz didn't generate files. Creating alternative visualizations...")
                    
                    # Create basic visualizations manually
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    categorical_cols = data.select_dtypes(include=['object']).columns
                    
                    if len(numeric_cols) > 0:
                        st.subheader("📊 Numeric Variables Distribution")
                        
                        # Create distribution plots
                        for col in numeric_cols[:4]:  # Limit to first 4 columns
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue')
                            ax.set_title(f'Distribution of {col}')
                            ax.set_xlabel(col)
                            ax.set_ylabel('Frequency')
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    if len(categorical_cols) > 0:
                        st.subheader("📊 Categorical Variables")
                        
                        # Create count plots
                        for col in categorical_cols[:3]:  # Limit to first 3 columns
                            if data[col].nunique() <= 10:  # Only if not too many categories
                                fig, ax = plt.subplots(figsize=(8, 4))
                                value_counts = data[col].value_counts().head(10)
                                ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
                                ax.set_title(f'Value Counts for {col}')
                                ax.set_xlabel(col)
                                ax.set_ylabel('Count')
                                ax.set_xticks(range(len(value_counts)))
                                ax.set_xticklabels(value_counts.index, rotation=45)
                                st.pyplot(fig)
                                plt.close(fig)
                    
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

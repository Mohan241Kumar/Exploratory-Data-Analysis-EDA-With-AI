import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
import tempfile
import os
import base64
import time
import gc
from autoviz.AutoViz_Class import AutoViz_Class
import warnings
warnings.filterwarnings('ignore')

# Fix numpy compatibility issue with Sweetviz
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = UserWarning

def safe_cleanup_temp_file(file_path, max_retries=5, delay=0.5):
    """
    Safely clean up temporary files with retry logic to handle Windows file locking issues
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                # Force garbage collection to release any file handles
                gc.collect()
                time.sleep(delay)
                os.unlink(file_path)
                return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                # Wait progressively longer between retries
                time.sleep(delay * (attempt + 1))
                continue
            else:
                st.warning(f"Could not clean up temporary file {file_path}: {str(e)}")
                st.info("The temporary file will be cleaned up by the system eventually.")
                return False
    return False

def create_comprehensive_visualizations(data):
    """Create comprehensive alternative visualizations when AutoViz fails"""
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Get column types
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 1. Dataset Overview Dashboard
    st.subheader("📊 Dataset Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{data.shape[1]:,}")
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{data.duplicated().sum():,}")
    
    # 2. Missing Values Heatmap
    if data.isnull().sum().sum() > 0:
        st.subheader("🔍 Missing Values Analysis")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.index, 
                y=missing_data.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # 3. Numeric Variables Analysis
    if len(numeric_cols) > 0:
        st.subheader("📈 Numeric Variables Analysis")
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            st.write("**Correlation Matrix**")
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix, 
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots for numeric columns
        st.write("**Distributions**")
        cols_per_row = 2
        for i in range(0, min(len(numeric_cols), 8), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig = px.histogram(
                        data, 
                        x=col_name, 
                        title=f'Distribution of {col_name}',
                        nbins=30
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # 4. Categorical Variables Analysis
    if len(categorical_cols) > 0:
        st.subheader("📊 Categorical Variables Analysis")
        
        for col_name in categorical_cols[:6]:  # Limit to first 6
            if data[col_name].nunique() <= 20:  # Only if reasonable number of categories
                st.write(f"**{col_name}**")
                value_counts = data[col_name].value_counts().head(15)
                
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f'Value Counts for {col_name}'
                )
                fig.update_layout(height=max(300, len(value_counts) * 25))
                st.plotly_chart(fig, use_container_width=True)
    
    # 5. Data Quality Summary
    st.subheader("🎯 Data Quality Summary")
    
    quality_metrics = []
    for col in data.columns:
        col_data = data[col]
        quality_metrics.append({
            'Column': col,
            'Type': str(col_data.dtype),
            'Missing %': f"{(col_data.isnull().sum() / len(col_data) * 100):.1f}%",
            'Unique Values': col_data.nunique(),
            'Most Frequent': str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else 'N/A'
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, use_container_width=True)

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
        temp_file_path = None
        try:
            # Create profile report
            profile = ProfileReport(
                data,
                title="Pandas Profiling Report",
                explorative=True,
                minimal=False
            )
            
            # Create temporary file with a unique name to avoid conflicts
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"pandas_profiling_report_{int(time.time() * 1000)}.html")
            
            # Generate the report
            profile.to_file(temp_file_path)
            
            # Wait a moment for file to be fully written
            time.sleep(1)
            
            # Read the HTML content with proper file handling
            html_content = None
            for attempt in range(3):
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    break
                except (OSError, PermissionError) as e:
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
                    else:
                        raise e
            
            if html_content:
                # Display in iframe
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Pandas Profiling Report",
                    data=html_content,
                    file_name="pandas_profiling_report.html",
                    mime="text/html"
                )
                
                st.success("✅ Pandas Profiling report generated successfully!")
            else:
                raise Exception("Could not read the generated report file")
                
        except Exception as e:
            st.error(f"Error generating Pandas Profiling report: {str(e)}")
            st.info("💡 This could be due to file access issues or data complexity. Try with a smaller dataset or check data types.")
            
        finally:
            # Clean up temporary file with retry logic
            if temp_file_path and os.path.exists(temp_file_path):
                # Wait a moment before cleanup to ensure all handles are released
                time.sleep(0.5)
                safe_cleanup_temp_file(temp_file_path)

# Sweetviz
if generate_sweetviz:
    st.header("🍯 Sweetviz Report")
    
    with st.spinner("Generating Sweetviz analysis..."):
        temp_file_path = None
        try:
            # Handle numpy compatibility issue
            if not hasattr(np, 'VisibleDeprecationWarning'):
                np.VisibleDeprecationWarning = UserWarning
            
            # Create Sweetviz report
            report = sv.analyze(data)
            
            # Create temporary file with a unique name to avoid conflicts
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"sweetviz_report_{int(time.time() * 1000)}.html")
            
            # Generate the report
            report.show_html(temp_file_path, open_browser=False)
            
            # Wait a moment for file to be fully written
            time.sleep(1)
            
            # Read the HTML content with proper file handling
            html_content = None
            for attempt in range(3):
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    break
                except (OSError, PermissionError) as e:
                    if attempt < 2:
                        time.sleep(0.5)
                        continue
                    else:
                        raise e
            
            if html_content:
                # Display in iframe
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Sweetviz Report",
                    data=html_content,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )
                
                st.success("✅ Sweetviz report generated successfully!")
            else:
                raise Exception("Could not read the generated report file")
                
        except Exception as e:
            st.error(f"Error generating Sweetviz report: {str(e)}")
            st.info("💡 This could be due to numpy compatibility or file access issues. Using alternative summary instead.")
            
        finally:
            # Clean up temporary file with retry logic
            if temp_file_path and os.path.exists(temp_file_path):
                # Wait a moment before cleanup to ensure all handles are released
                time.sleep(0.5)
                safe_cleanup_temp_file(temp_file_path)
            
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
            # Convert data types to strings to avoid JSON serialization issues
            type_names = [str(dtype) for dtype in type_counts.index]
            fig = px.pie(values=type_counts.values, names=type_names, title="Data Types Distribution")
            st.plotly_chart(fig, use_container_width=True)

# AutoViz
if generate_autoviz:
    st.header("🎨 AutoViz Report")
    
    with st.spinner("Generating AutoViz visualizations..."):
        try:
            # Import required libraries
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Data preprocessing for AutoViz
            st.info("🔄 Preprocessing data for AutoViz...")
            
            # Clean data for AutoViz compatibility
            viz_data = data.copy()
            
            # Handle missing values (AutoViz can be sensitive to NaN)
            for col in viz_data.columns:
                if viz_data[col].dtype == 'object':
                    viz_data[col] = viz_data[col].fillna('Unknown')
                else:
                    viz_data[col] = viz_data[col].fillna(viz_data[col].median() if viz_data[col].notna().any() else 0)
            
            # Limit data size for better performance
            if viz_data.shape[0] > 5000:
                viz_data = viz_data.sample(n=5000, random_state=42)
                st.info(f"📊 Sampled 5000 rows from {data.shape[0]} for visualization")
            
            # Limit columns for better performance
            if viz_data.shape[1] > 20:
                # Prioritize numeric columns and first few categorical
                numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = viz_data.select_dtypes(include=['object']).columns.tolist()
                selected_cols = numeric_cols[:15] + categorical_cols[:5]
                viz_data = viz_data[selected_cols]
                st.info(f"📊 Using {len(selected_cols)} columns out of {data.shape[1]} for visualization")
            
            # Create AutoViz instance
            av = AutoViz_Class()
            
            # Create a temporary directory for plots
            temp_dir = tempfile.mkdtemp(prefix="autoviz_")
            
            try:
                st.info("🎨 Running AutoViz analysis...")
                
                # Multiple AutoViz attempts with different configurations
                plot_files = []
                
                # Attempt 1: Standard configuration
                try:
                    dfte = av.AutoViz(
                        filename='',
                        sep=',',
                        depVar='',
                        dfte=viz_data,
                        header=0,
                        verbose=1,
                        lowess=False,
                        chart_format='png',
                        max_rows_analyzed=min(2000, viz_data.shape[0]),
                        max_cols_analyzed=min(30, viz_data.shape[1]),
                        save_plot_dir=temp_dir
                    )
                    
                    # Wait for files to be generated
                    time.sleep(2)
                    
                except Exception as e:
                    st.warning(f"Standard AutoViz failed: {str(e)[:100]}...")
                
                # Attempt 2: Minimal configuration if first attempt failed
                if not any(os.listdir(temp_dir)):
                    try:
                        st.info("🔄 Trying minimal AutoViz configuration...")
                        dfte = av.AutoViz(
                            filename='',
                            sep=',',
                            depVar='',
                            dfte=viz_data,
                            header=0,
                            verbose=0,
                            lowess=False,
                            chart_format='png',
                            max_rows_analyzed=1000,
                            max_cols_analyzed=10,
                            save_plot_dir=temp_dir
                        )
                        time.sleep(2)
                    except Exception as e:
                        st.warning(f"Minimal AutoViz also failed: {str(e)[:100]}...")
                
                # Look for generated plots
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.svg', '.jpg', '.jpeg')):
                            plot_files.append(os.path.join(root, file))
                
                if plot_files:
                    st.success(f"✅ Generated {len(plot_files)} AutoViz visualizations!")
                    
                    # Display plots
                    for i, plot_path in enumerate(plot_files):
                        if os.path.exists(plot_path):
                            plot_name = os.path.basename(plot_path).replace('.png', '').replace('.svg', '').replace('.jpg', '').replace('.jpeg', '')
                            st.subheader(f"📊 {plot_name}")
                            
                            try:
                                st.image(plot_path, use_column_width=True)
                            except Exception as e:
                                st.error(f"Could not display {plot_name}: {str(e)}")
                
                else:
                    # AutoViz didn't generate files - provide comprehensive alternatives
                    st.warning("⚠️ AutoViz didn't generate visualization files.")
                    st.info("🎨 Creating comprehensive alternative visualizations...")
                    
                    # Enhanced alternative visualizations
                    create_comprehensive_visualizations(data)
                        
            finally:
                # Clean up temporary directory
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"Error with AutoViz: {str(e)}")
            st.info("🎨 Creating fallback visualizations...")
            create_comprehensive_visualizations(data)



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
        'Data Type': [str(dtype) for dtype in dtype_counts.index],
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

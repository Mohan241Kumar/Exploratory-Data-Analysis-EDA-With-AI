import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from openai import OpenAI
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Auto Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Dashboard Generator")
st.markdown("AI-powered dashboard creation similar to Power BI")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    st.error("OpenAI API key not found or invalid. Please check your configuration.")
    st.stop()

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

def generate_dashboard_config(data_info):
    """Generate dashboard configuration using AI"""
    
    prompt = f"""
    You are a business intelligence expert. Create an optimal dashboard configuration for this dataset.
    
    Dataset Info:
    - Shape: {data_info['shape']}
    - Columns: {data_info['columns']}
    - Data Types: {data_info['dtypes']}
    - Sample Data: {data_info['sample_data']}
    
    Create a comprehensive dashboard with:
    1. Key Performance Indicators (KPIs)
    2. Trend charts
    3. Distribution charts
    4. Comparison charts
    5. Correlation analysis
    6. Geographic visualization (if applicable)
    
    Respond in JSON format:
    {{
        "dashboard_title": "Descriptive title for the dashboard",
        "kpis": [
            {{
                "name": "KPI Name",
                "column": "column_name",
                "aggregation": "sum/mean/count/max/min",
                "description": "What this KPI measures"
            }}
        ],
        "charts": [
            {{
                "title": "Chart Title",
                "type": "bar/line/pie/scatter/heatmap/box",
                "x_column": "column_name",
                "y_column": "column_name",
                "color_column": "column_name or null",
                "description": "What this chart shows"
            }}
        ],
        "insights": [
            "Key insight 1",
            "Key insight 2"
        ],
        "layout": {{
            "rows": 3,
            "columns": 2
        }}
    }}
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating dashboard config: {str(e)}")
        return None

def create_kpi_card(title, value, description, delta=None):
    """Create a KPI card"""
    
    # Format large numbers
    if isinstance(value, (int, float)):
        if value >= 1000000:
            formatted_value = f"{value/1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"{value/1000:.1f}K"
        else:
            formatted_value = f"{value:.1f}"
    else:
        formatted_value = str(value)
    
    with st.container():
        st.metric(
            label=title,
            value=formatted_value,
            delta=delta,
            help=description
        )

def create_chart(chart_config, data):
    """Create a chart based on configuration"""
    
    chart_type = chart_config.get('type', 'bar')
    title = chart_config.get('title', 'Chart')
    x_col = chart_config.get('x_column')
    y_col = chart_config.get('y_column')
    color_col = chart_config.get('color_column')
    
    try:
        if chart_type == 'bar':
            if x_col and y_col and x_col in data.columns and y_col in data.columns:
                fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
            elif x_col and x_col in data.columns:
                # Value counts for categorical data
                if data[x_col].dtype == 'object':
                    value_counts = data[x_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=title)
                    fig.update_xaxis(title=x_col)
                    fig.update_yaxis(title='Count')
                else:
                    fig = px.histogram(data, x=x_col, title=title)
            else:
                return None
                
        elif chart_type == 'line':
            if x_col and y_col and x_col in data.columns and y_col in data.columns:
                fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title)
            else:
                return None
                
        elif chart_type == 'pie':
            if x_col and x_col in data.columns:
                value_counts = data[x_col].value_counts().head(8)
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
            else:
                return None
                
        elif chart_type == 'scatter':
            if x_col and y_col and x_col in data.columns and y_col in data.columns:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
            else:
                return None
                
        elif chart_type == 'box':
            if y_col and y_col in data.columns:
                fig = px.box(data, y=y_col, x=x_col, title=title)
            else:
                return None
                
        elif chart_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, title=title)
            else:
                return None
                
        else:
            return None
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# Sidebar controls
st.sidebar.header("🎛️ Dashboard Controls")

dashboard_type = st.sidebar.selectbox(
    "Dashboard Type",
    ["AI Generated", "Business Analytics", "Data Quality", "Custom"]
)

if st.sidebar.button("🚀 Generate Dashboard", type="primary"):
    with st.spinner("AI is creating your dashboard..."):
        # Prepare data info for AI
        data_info = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample_data': data.head(3).to_dict()
        }
        
        # Generate dashboard configuration
        dashboard_config = generate_dashboard_config(data_info)
        
        if dashboard_config:
            st.session_state.dashboard_config = dashboard_config
            st.success("Dashboard generated successfully!")

# Display dashboard
if 'dashboard_config' in st.session_state:
    config = st.session_state.dashboard_config
    
    # Dashboard title
    st.header(config.get('dashboard_title', 'Data Dashboard'))
    
    # KPIs section
    if config.get('kpis'):
        st.subheader("📊 Key Performance Indicators")
        
        kpi_cols = st.columns(min(len(config['kpis']), 4))
        
        for i, kpi in enumerate(config['kpis']):
            col_name = kpi.get('column')
            aggregation = kpi.get('aggregation', 'count')
            
            if col_name and col_name in data.columns:
                if aggregation == 'sum':
                    value = data[col_name].sum()
                elif aggregation == 'mean':
                    value = data[col_name].mean()
                elif aggregation == 'count':
                    value = data[col_name].count()
                elif aggregation == 'max':
                    value = data[col_name].max()
                elif aggregation == 'min':
                    value = data[col_name].min()
                else:
                    value = data[col_name].count()
                
                with kpi_cols[i % 4]:
                    create_kpi_card(
                        kpi.get('name', col_name),
                        value,
                        kpi.get('description', f'{aggregation.title()} of {col_name}')
                    )
    
    # Charts section
    if config.get('charts'):
        st.subheader("📈 Analytics Charts")
        
        # Create chart grid
        layout = config.get('layout', {'rows': 2, 'columns': 2})
        charts_per_row = layout.get('columns', 2)
        
        chart_rows = []
        current_row = []
        
        for i, chart_config in enumerate(config['charts']):
            current_row.append(chart_config)
            
            if len(current_row) == charts_per_row or i == len(config['charts']) - 1:
                chart_rows.append(current_row)
                current_row = []
        
        for row in chart_rows:
            cols = st.columns(len(row))
            
            for i, chart_config in enumerate(row):
                with cols[i]:
                    fig = create_chart(chart_config, data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Could not create chart: {chart_config.get('title', 'Unknown')}")
    
    # Insights section
    if config.get('insights'):
        st.subheader("💡 AI Insights")
        
        for insight in config['insights']:
            st.info(f"🔍 {insight}")
    
    # Data summary section
    st.subheader("📊 Data Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        st.metric("Total Rows", f"{data.shape[0]:,}")
    
    with summary_cols[1]:
        st.metric("Total Columns", f"{data.shape[1]:,}")
    
    with summary_cols[2]:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    
    with summary_cols[3]:
        st.metric("Data Types", f"{data.dtypes.nunique()}")

# Manual dashboard builder
st.header("🛠️ Custom Dashboard Builder")

with st.expander("Create Custom Charts"):
    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Heatmap"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_column = st.selectbox("X-axis", ["None"] + data.columns.tolist())
    
    with col2:
        y_column = st.selectbox("Y-axis", ["None"] + data.columns.tolist())
    
    with col3:
        color_column = st.selectbox("Color by", ["None"] + data.columns.tolist())
    
    if st.button("Create Custom Chart"):
        custom_config = {
            'type': chart_type.lower().replace(' ', ''),
            'title': f'Custom {chart_type}',
            'x_column': x_column if x_column != "None" else None,
            'y_column': y_column if y_column != "None" else None,
            'color_column': color_column if color_column != "None" else None
        }
        
        fig = create_chart(custom_config, data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Advanced analytics
st.header("🔬 Advanced Analytics")

analytics_tabs = st.tabs(["Correlation Analysis", "Distribution Analysis", "Time Series", "Geographic"])

with analytics_tabs[0]:
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.subheader("Top Correlations")
        correlation_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(correlation_pairs)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(corr_df.head(10), use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

with analytics_tabs[1]:
    st.subheader("📊 Distribution Analysis")
    
    if numeric_cols.any():
        selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(data, x=selected_col, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(data, y=selected_col, title=f'Box Plot of {selected_col}')
            st.plotly_chart(fig, use_container_width=True)

with analytics_tabs[2]:
    st.subheader("📈 Time Series Analysis")
    
    # Detect date columns
    date_cols = []
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                pd.to_datetime(data[col], errors='raise')
                date_cols.append(col)
            except:
                pass
    
    if date_cols:
        date_col = st.selectbox("Select date column:", date_cols)
        value_col = st.selectbox("Select value column:", numeric_cols)
        
        if date_col and value_col:
            # Convert to datetime
            data_copy = data.copy()
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
            
            # Create time series plot
            fig = px.line(data_copy, x=date_col, y=value_col, title=f'{value_col} over Time')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date columns detected for time series analysis.")

with analytics_tabs[3]:
    st.subheader("🗺️ Geographic Analysis")
    
    # Look for geographic columns
    geo_cols = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in ['lat', 'lon', 'city', 'state', 'country', 'zip', 'postal']):
            geo_cols.append(col)
    
    if geo_cols:
        st.write("Detected geographic columns:")
        for col in geo_cols:
            st.write(f"• {col}")
        
        # Try to create a map if lat/lon are available
        lat_cols = [col for col in geo_cols if 'lat' in col.lower()]
        lon_cols = [col for col in geo_cols if 'lon' in col.lower()]
        
        if lat_cols and lon_cols:
            lat_col = st.selectbox("Latitude column:", lat_cols)
            lon_col = st.selectbox("Longitude column:", lon_cols)
            
            if lat_col and lon_col:
                fig = px.scatter_mapbox(
                    data,
                    lat=lat_col,
                    lon=lon_col,
                    hover_data=data.columns.tolist(),
                    zoom=3,
                    height=600,
                    title="Geographic Distribution"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geographic columns detected.")

# Export dashboard
st.header("📥 Export Dashboard")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Dashboard Config"):
        if 'dashboard_config' in st.session_state:
            config_json = json.dumps(st.session_state.dashboard_config, indent=2)
            st.download_button(
                label="📥 Download Config JSON",
                data=config_json,
                file_name="dashboard_config.json",
                mime="application/json"
            )

with col2:
    if st.button("📈 Generate Report"):
        if 'dashboard_config' in st.session_state:
            report = {
                'dashboard_title': st.session_state.dashboard_config.get('dashboard_title', 'Dashboard'),
                'generated_at': datetime.now().isoformat(),
                'data_summary': {
                    'rows': data.shape[0],
                    'columns': data.shape[1],
                    'missing_values': data.isnull().sum().sum(),
                    'data_types': data.dtypes.value_counts().to_dict()
                },
                'insights': st.session_state.dashboard_config.get('insights', [])
            }
            
            st.download_button(
                label="📥 Download Report",
                data=json.dumps(report, indent=2, default=str),
                file_name="dashboard_report.json",
                mime="application/json"
            )

st.markdown("---")
st.markdown("**Automated Dashboard** - AI-powered business intelligence visualization")
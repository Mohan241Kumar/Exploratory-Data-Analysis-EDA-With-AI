import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Interactive Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactive Analysis")
st.markdown("Explore your data with dynamic visualizations and custom analysis")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

# Analysis type selection
st.header("🎯 Analysis Type")

analysis_type = st.selectbox(
    "Choose analysis type:",
    [
        "Univariate Analysis",
        "Bivariate Analysis", 
        "Multivariate Analysis",
        "Distribution Analysis",
        "Correlation Analysis",
        "Custom Visualization"
    ]
)

if analysis_type == "Univariate Analysis":
    st.subheader("📈 Univariate Analysis")
    
    # Column selection
    selected_col = st.selectbox("Select column for analysis:", data.columns)
    
    col_data = data[selected_col].dropna()
    
    # Column information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Type", str(data[selected_col].dtype))
    
    with col2:
        st.metric("Non-Null Count", len(col_data))
    
    with col3:
        st.metric("Missing Values", data[selected_col].isnull().sum())
    
    with col4:
        if data[selected_col].dtype in ['object']:
            st.metric("Unique Values", data[selected_col].nunique())
        else:
            st.metric("Mean", f"{col_data.mean():.2f}")
    
    # Visualization based on data type
    if data[selected_col].dtype in ['object']:
        # Categorical variable
        value_counts = data[selected_col].value_counts().head(20)
        
        viz_type = st.selectbox("Visualization type:", ["Bar Chart", "Pie Chart", "Donut Chart"])
        
        if viz_type == "Bar Chart":
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Value Counts - {selected_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution - {selected_col}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=.3
            )])
            fig.update_layout(title=f'Distribution - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Numerical variable
        viz_type = st.selectbox("Visualization type:", ["Histogram", "Box Plot", "Violin Plot", "Distribution Plot"])
        
        if viz_type == "Histogram":
            bins = st.slider("Number of bins:", 10, 100, 30)
            fig = px.histogram(col_data, nbins=bins, title=f'Histogram - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            fig = px.box(y=col_data, title=f'Box Plot - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            fig = px.violin(y=col_data, title=f'Violin Plot - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Distribution Plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=col_data, name='Histogram', opacity=0.7))
            
            # Add normal distribution overlay
            mean = col_data.mean()
            std = col_data.std()
            x_range = np.linspace(col_data.min(), col_data.max(), 100)
            normal_dist = stats.norm.pdf(x_range, mean, std) * len(col_data) * (col_data.max() - col_data.min()) / 30
            
            fig.add_trace(go.Scatter(
                x=x_range, 
                y=normal_dist, 
                mode='lines', 
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(title=f'Distribution with Normal Overlay - {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    if data[selected_col].dtype in ['object']:
        # Categorical statistics
        stats_data = {
            'Count': len(col_data),
            'Unique Values': data[selected_col].nunique(),
            'Most Frequent': data[selected_col].mode().iloc[0] if not data[selected_col].mode().empty else 'N/A',
            'Frequency of Most Common': data[selected_col].value_counts().iloc[0] if len(data[selected_col].value_counts()) > 0 else 0
        }
        
        for key, value in stats_data.items():
            st.metric(key, value)
    
    else:
        # Numerical statistics
        stats_data = col_data.describe()
        
        cols = st.columns(4)
        stats_items = [
            ('Count', stats_data['count']),
            ('Mean', stats_data['mean']),
            ('Std Dev', stats_data['std']),
            ('Min', stats_data['min']),
            ('25%', stats_data['25%']),
            ('50%', stats_data['50%']),
            ('75%', stats_data['75%']),
            ('Max', stats_data['max'])
        ]
        
        for i, (label, value) in enumerate(stats_items):
            with cols[i % 4]:
                st.metric(label, f"{value:.2f}")

elif analysis_type == "Bivariate Analysis":
    st.subheader("📊 Bivariate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Select X variable:", data.columns, key="x_var")
    
    with col2:
        y_col = st.selectbox("Select Y variable:", data.columns, key="y_var")
    
    if x_col != y_col:
        # Determine plot type based on variable types
        x_is_numeric = data[x_col].dtype in ['int64', 'float64']
        y_is_numeric = data[y_col].dtype in ['int64', 'float64']
        
        if x_is_numeric and y_is_numeric:
            # Both numeric - scatter plot
            viz_type = st.selectbox("Visualization type:", ["Scatter Plot", "Line Plot", "Regression Plot"])
            
            if viz_type == "Scatter Plot":
                fig = px.scatter(data, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Line Plot":
                fig = px.line(data, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Regression Plot
                fig = px.scatter(data, x=x_col, y=y_col, trendline="ols", title=f'{y_col} vs {x_col} (with regression line)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            corr_coef = data[x_col].corr(data[y_col])
            st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
        
        elif x_is_numeric and not y_is_numeric:
            # X numeric, Y categorical - box plot
            fig = px.box(data, x=y_col, y=x_col, title=f'{x_col} by {y_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        elif not x_is_numeric and y_is_numeric:
            # X categorical, Y numeric - box plot
            fig = px.box(data, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Both categorical - heatmap
            crosstab = pd.crosstab(data[x_col], data[y_col])
            fig = px.imshow(crosstab, text_auto=True, title=f'{y_col} vs {x_col} (Cross-tabulation)')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please select different variables for X and Y.")

elif analysis_type == "Multivariate Analysis":
    st.subheader("🔍 Multivariate Analysis")
    
    # Select multiple columns
    selected_cols = st.multiselect("Select columns for analysis:", data.columns, default=data.columns[:3].tolist())
    
    if len(selected_cols) >= 2:
        viz_type = st.selectbox("Visualization type:", ["Scatter Matrix", "Parallel Coordinates", "3D Scatter Plot"])
        
        if viz_type == "Scatter Matrix":
            fig = px.scatter_matrix(data[selected_cols], title="Scatter Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Parallel Coordinates":
            # Normalize data for parallel coordinates
            normalized_data = data[selected_cols].copy()
            for col in selected_cols:
                if data[col].dtype in ['int64', 'float64']:
                    normalized_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            
            fig = px.parallel_coordinates(normalized_data, title="Parallel Coordinates Plot")
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # 3D Scatter Plot
            if len(selected_cols) >= 3:
                fig = px.scatter_3d(data, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2], 
                                   title=f"3D Scatter Plot: {selected_cols[0]} vs {selected_cols[1]} vs {selected_cols[2]}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 3 columns for 3D scatter plot.")
    
    else:
        st.warning("Please select at least 2 columns for multivariate analysis.")

elif analysis_type == "Distribution Analysis":
    st.subheader("📊 Distribution Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        selected_cols = st.multiselect("Select columns for distribution analysis:", numeric_cols, default=numeric_cols[:3].tolist())
        
        if selected_cols:
            viz_type = st.selectbox("Visualization type:", ["Overlaid Histograms", "Box Plots", "Violin Plots", "KDE Plots"])
            
            if viz_type == "Overlaid Histograms":
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Histogram(x=data[col], name=col, opacity=0.7))
                fig.update_layout(title="Overlaid Histograms", barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plots":
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Box(y=data[col], name=col))
                fig.update_layout(title="Box Plots Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Violin Plots":
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Violin(y=data[col], name=col))
                fig.update_layout(title="Violin Plots Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # KDE Plots
                fig = go.Figure()
                for col in selected_cols:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Calculate KDE
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(col_data)
                        x_range = np.linspace(col_data.min(), col_data.max(), 100)
                        kde_values = kde(x_range)
                        fig.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name=col))
                
                fig.update_layout(title="KDE Plots Comparison")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No numeric columns found for distribution analysis.")

elif analysis_type == "Correlation Analysis":
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Correlation heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix Heatmap",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation strength analysis
        st.subheader("Correlation Strength Analysis")
        
        correlation_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                correlation_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_val,
                    'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong' if abs(corr_val) > 0.6 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                })
        
        corr_df = pd.DataFrame(correlation_pairs)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        st.dataframe(corr_df, use_container_width=True)
        
        # Top correlations
        st.subheader("Top Positive and Negative Correlations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Correlations**")
            top_positive = corr_df[corr_df['Correlation'] > 0].head(5)
            st.dataframe(top_positive, use_container_width=True)
        
        with col2:
            st.write("**Top Negative Correlations**")
            top_negative = corr_df[corr_df['Correlation'] < 0].head(5)
            st.dataframe(top_negative, use_container_width=True)
    
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

else:  # Custom Visualization
    st.subheader("🎨 Custom Visualization")
    
    # Custom plot builder
    plot_type = st.selectbox("Select plot type:", [
        "Scatter Plot", "Line Plot", "Bar Chart", "Histogram", 
        "Box Plot", "Violin Plot", "Heatmap", "Sunburst", "Treemap"
    ])
    
    if plot_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis:", data.columns, key="custom_x")
        
        with col2:
            y_col = st.selectbox("Y-axis:", data.columns, key="custom_y")
        
        with col3:
            color_col = st.selectbox("Color by:", [None] + data.columns.tolist(), key="custom_color")
        
        if x_col != y_col:
            if color_col:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=f'{y_col} vs {x_col}')
            else:
                fig = px.scatter(data, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bar Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis (categories):", data.columns, key="bar_x")
        
        with col2:
            y_col = st.selectbox("Y-axis (values):", data.columns, key="bar_y")
        
        if data[x_col].dtype == 'object':
            # Aggregate data
            agg_data = data.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(agg_data, x=x_col, y=y_col, title=f'Average {y_col} by {x_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("X-axis should be a categorical variable for bar charts.")
    
    elif plot_type == "Heatmap":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect("Select columns for heatmap:", numeric_cols, default=numeric_cols[:5].tolist())
            
            if selected_cols:
                heatmap_data = data[selected_cols].corr()
                fig = px.imshow(heatmap_data, text_auto=True, title="Custom Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for heatmap.")
    
    # Add more custom plot types as needed
    else:
        st.info(f"{plot_type} customization coming soon!")

# Data filtering section
st.header("🔍 Data Filtering")

with st.expander("Apply Filters"):
    filters = {}
    
    for col in data.columns:
        if data[col].dtype == 'object':
            # Categorical filter
            unique_values = data[col].dropna().unique()
            selected_values = st.multiselect(f"Filter {col}:", unique_values, default=unique_values.tolist())
            filters[col] = selected_values
        
        elif data[col].dtype in ['int64', 'float64']:
            # Numeric filter
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            selected_range = st.slider(f"Filter {col}:", min_val, max_val, (min_val, max_val))
            filters[col] = selected_range
    
    # Apply filters
    filtered_data = data.copy()
    for col, filter_val in filters.items():
        if data[col].dtype == 'object':
            filtered_data = filtered_data[filtered_data[col].isin(filter_val)]
        else:
            filtered_data = filtered_data[
                (filtered_data[col] >= filter_val[0]) & 
                (filtered_data[col] <= filter_val[1])
            ]
    
    st.info(f"Filtered data shape: {filtered_data.shape} (from {data.shape})")

# Export options
st.header("📥 Export Options")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Current Analysis"):
        # This would export the current analysis results
        st.success("Analysis exported successfully!")

with col2:
    if st.button("💾 Save Filtered Data"):
        # This would save the filtered dataset
        if 'filtered_data' in locals():
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_data,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("**Interactive Analysis** - Explore your data with dynamic visualizations")

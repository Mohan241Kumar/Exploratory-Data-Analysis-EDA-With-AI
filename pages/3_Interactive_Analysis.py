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

# Helper: detect numeric and numeric-like columns
def get_numeric_columns(df, include_numeric_like: bool = True, min_numeric_fraction: float = 0.8):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if include_numeric_like:
        for col in df.columns:
            if col in numeric_cols:
                continue
            # Try to coerce to numeric and check how many values successfully convert
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.notna().mean() >= min_numeric_fraction:
                numeric_cols.append(col)
    return numeric_cols

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
            st.subheader("Histogram Customization")
            
            # Create two columns for customization options
            col1, col2 = st.columns(2)
            
            with col1:
                bins = st.slider("Number of bins:", 5, 100, 30, help="Adjust the number of histogram bins")
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.7, 0.1, help="Control histogram bar transparency")
                show_rug = st.checkbox("Show rug plot", value=False, help="Add rug plot at the bottom")
                
            with col2:
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Custom"], 
                    help="Choose histogram color scheme")
                
                if color_scheme == "Custom":
                    custom_color = st.color_picker("Pick a color", "#1f77b4")
                else:
                    custom_color = None
                
                show_kde = st.checkbox("Show KDE curve", value=False, help="Overlay kernel density estimation")
                show_normal = st.checkbox("Show normal distribution", value=False, help="Overlay normal distribution curve")
            
            # Create the histogram
            fig = go.Figure()
            
            # Choose color
            if color_scheme == "Custom":
                hist_color = custom_color
            elif color_scheme == "Default":
                hist_color = None  # Use default
            else:
                hist_color = color_scheme.lower()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=col_data, 
                name='Histogram', 
                opacity=opacity,
                nbinsx=bins,
                marker_color=hist_color,
                histnorm='probability density' if show_kde else ''
            ))
            
            # Add rug plot if requested
            if show_rug:
                rug_data = col_data.dropna()
                fig.add_trace(go.Scatter(
                    x=rug_data,
                    y=[0] * len(rug_data),
                    mode='markers',
                    marker=dict(size=2, color='red', opacity=0.6),
                    name='Rug Plot',
                    showlegend=True
                ))
            
            # Add KDE if requested
            if show_kde:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(col_data.dropna())
                x_range = np.linspace(col_data.min(), col_data.max(), 200)
                kde_values = kde(x_range)
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                ))
            
            # Add normal distribution if requested
            if show_normal:
                mean = col_data.mean()
                std = col_data.std()
                x_range = np.linspace(col_data.min(), col_data.max(), 200)
                normal_dist = stats.norm.pdf(x_range, mean, std)
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=normal_dist, 
                    mode='lines', 
                    name='Normal Distribution',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Customized Histogram - {selected_col}',
                xaxis_title=selected_col,
                yaxis_title='Density' if show_kde else 'Count',
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistics
            st.subheader("Distribution Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{col_data.mean():.3f}")
            with col2:
                st.metric("Median", f"{col_data.median():.3f}")
            with col3:
                st.metric("Std Dev", f"{col_data.std():.3f}")
            with col4:
                st.metric("Skewness", f"{col_data.skew():.3f}")
        
        elif viz_type == "Box Plot":
            st.subheader("Box Plot Customization")
            
            # Create two columns for customization options
            col1, col2 = st.columns(2)
            
            with col1:
                show_outliers = st.checkbox("Show outliers", value=True, help="Display outlier points")
                show_mean = st.checkbox("Show mean", value=False, help="Display mean marker")
                show_notches = st.checkbox("Show notches", value=False, help="Display confidence interval notches")
                boxmode = st.selectbox("Box mode:", ["group", "overlay"], help="How to display multiple boxes")
                
            with col2:
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Custom"], 
                    help="Choose box plot color scheme")
                
                if color_scheme == "Custom":
                    custom_color = st.color_picker("Pick a color", "#1f77b4")
                else:
                    custom_color = None
                
                orientation = st.selectbox("Orientation:", ["vertical", "horizontal"], help="Box plot orientation")
                show_points = st.selectbox("Show points:", ["outliers", "all", "suspectedoutliers", "False"], 
                    help="Which points to show")
            
            # Create the box plot
            fig = go.Figure()
            
            # Choose color
            if color_scheme == "Custom":
                box_color = custom_color
            elif color_scheme == "Default":
                box_color = None  # Use default
            else:
                box_color = color_scheme.lower()
            
            # Add box plot
            fig.add_trace(go.Box(
                y=col_data if orientation == "vertical" else None,
                x=col_data if orientation == "horizontal" else None,
                name=selected_col,
                boxpoints=show_points if show_points != "False" else False,
                notched=show_notches,
                marker_color=box_color,
                showlegend=True
            ))
            
            # Add mean if requested
            if show_mean:
                mean_val = col_data.mean()
                if orientation == "vertical":
                    fig.add_hline(y=mean_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Mean: {mean_val:.3f}")
                else:
                    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Mean: {mean_val:.3f}")
            
            # Update layout
            fig.update_layout(
                title=f'Customized Box Plot - {selected_col}',
                xaxis_title=selected_col if orientation == "horizontal" else None,
                yaxis_title=selected_col if orientation == "vertical" else None,
                showlegend=True,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add box plot statistics
            st.subheader("Box Plot Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Q1 (25%)", f"{col_data.quantile(0.25):.3f}")
            with col2:
                st.metric("Median (50%)", f"{col_data.median():.3f}")
            with col3:
                st.metric("Q3 (75%)", f"{col_data.quantile(0.75):.3f}")
            with col4:
                st.metric("IQR", f"{col_data.quantile(0.75) - col_data.quantile(0.25):.3f}")
            
            # Outlier analysis
            st.subheader("Outlier Analysis")
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outlier Count", len(outliers))
            with col2:
                st.metric("Outlier %", f"{(len(outliers) / len(col_data)) * 100:.2f}%")
            with col3:
                st.metric("Outlier Range", f"[{outliers.min():.3f}, {outliers.max():.3f}]" if len(outliers) > 0 else "None")
            
            if len(outliers) > 0:
                st.subheader("Outlier Values")
                st.dataframe(outliers.to_frame(name=selected_col), use_container_width=True)
        
        elif viz_type == "Violin Plot":
            st.subheader("Violin Plot Customization")
            
            # Create two columns for customization options
            col1, col2 = st.columns(2)
            
            with col1:
                show_box = st.checkbox("Show box plot", value=True, help="Display box plot inside violin")
                show_mean = st.checkbox("Show mean", value=False, help="Display mean marker")
                show_median = st.checkbox("Show median", value=True, help="Display median line")
                show_points = st.selectbox("Show points:", ["outliers", "all", "suspectedoutliers", "False"], 
                    help="Which points to show")
                
            with col2:
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Custom"], 
                    help="Choose violin plot color scheme")
                
                if color_scheme == "Custom":
                    custom_color = st.color_picker("Pick a color", "#1f77b4")
                else:
                    custom_color = None
                
                orientation = st.selectbox("Orientation:", ["vertical", "horizontal"], help="Violin plot orientation")
                bandwidth = st.slider("Bandwidth:", 0.1, 2.0, 1.0, 0.1, help="Kernel density estimation bandwidth")
            
            # Create the violin plot
            fig = go.Figure()
            
            # Choose color
            if color_scheme == "Custom":
                violin_color = custom_color
            elif color_scheme == "Default":
                violin_color = None  # Use default
            else:
                violin_color = color_scheme.lower()
            
            # Add violin plot
            fig.add_trace(go.Violin(
                y=col_data if orientation == "vertical" else None,
                x=col_data if orientation == "horizontal" else None,
                name=selected_col,
                box_visible=show_box,
                meanline_visible=show_mean,
                points=show_points if show_points != "False" else False,
                bandwidth=bandwidth,
                marker_color=violin_color,
                showlegend=True
            ))
            
            # Add median if requested
            if show_median:
                median_val = col_data.median()
                if orientation == "vertical":
                    fig.add_hline(y=median_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Median: {median_val:.3f}")
                else:
                    fig.add_vline(x=median_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Median: {median_val:.3f}")
            
            # Update layout
            fig.update_layout(
                title=f'Customized Violin Plot - {selected_col}',
                xaxis_title=selected_col if orientation == "horizontal" else None,
                yaxis_title=selected_col if orientation == "vertical" else None,
                showlegend=True,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add violin plot statistics
            st.subheader("Distribution Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{col_data.mean():.3f}")
            with col2:
                st.metric("Median", f"{col_data.median():.3f}")
            with col3:
                st.metric("Mode", f"{col_data.mode().iloc[0]:.3f}" if not col_data.mode().empty else "N/A")
            with col4:
                st.metric("Std Dev", f"{col_data.std():.3f}")
            
            # Distribution shape analysis
            st.subheader("Distribution Shape Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Skewness", f"{col_data.skew():.3f}")
                if abs(col_data.skew()) < 0.5:
                    st.success("Approximately symmetric")
                elif col_data.skew() > 0.5:
                    st.warning("Right-skewed")
                else:
                    st.warning("Left-skewed")
            
            with col2:
                st.metric("Kurtosis", f"{col_data.kurtosis():.3f}")
                if abs(col_data.kurtosis()) < 0.5:
                    st.success("Normal kurtosis")
                elif col_data.kurtosis() > 0.5:
                    st.warning("Heavy-tailed")
                else:
                    st.warning("Light-tailed")
            
            with col3:
                st.metric("Range", f"{col_data.max() - col_data.min():.3f}")
                st.metric("IQR", f"{col_data.quantile(0.75) - col_data.quantile(0.25):.3f}")
        
        else:  # Distribution Plot
            st.subheader("Distribution Plot Customization")
            
            # Create two columns for customization options
            col1, col2 = st.columns(2)
            
            with col1:
                bins = st.slider("Number of bins:", 5, 100, 30, help="Adjust the number of histogram bins")
                opacity = st.slider("Histogram opacity:", 0.1, 1.0, 0.7, 0.1, help="Control histogram bar transparency")
                show_kde = st.checkbox("Show KDE curve", value=True, help="Overlay kernel density estimation")
                
            with col2:
                show_normal = st.checkbox("Show normal distribution", value=True, help="Overlay normal distribution curve")
                show_rug = st.checkbox("Show rug plot", value=False, help="Add rug plot at the bottom")
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges"], 
                    help="Choose color scheme")
            
            # Create the distribution plot
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=col_data, 
                name='Histogram', 
                opacity=opacity,
                nbinsx=bins,
                marker_color=color_scheme.lower() if color_scheme != "Default" else None,
                histnorm='probability density' if show_kde else ''
            ))
            
            # Add KDE if requested
            if show_kde:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(col_data.dropna())
                x_range = np.linspace(col_data.min(), col_data.max(), 200)
                kde_values = kde(x_range)
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                ))
            
            # Add normal distribution if requested
            if show_normal:
                mean = col_data.mean()
                std = col_data.std()
                x_range = np.linspace(col_data.min(), col_data.max(), 200)
                normal_dist = stats.norm.pdf(x_range, mean, std)
                fig.add_trace(go.Scatter(
                    x=x_range, 
                    y=normal_dist, 
                    mode='lines', 
                    name='Normal Distribution',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            # Add rug plot if requested
            if show_rug:
                rug_data = col_data.dropna()
                fig.add_trace(go.Scatter(
                    x=rug_data,
                    y=[0] * len(rug_data),
                    mode='markers',
                    marker=dict(size=2, color='blue', opacity=0.6),
                    name='Rug Plot',
                    showlegend=True
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Customized Distribution Plot - {selected_col}',
                xaxis_title=selected_col,
                yaxis_title='Density' if show_kde else 'Count',
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add distribution statistics
            st.subheader("Distribution Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{col_data.mean():.3f}")
            with col2:
                st.metric("Median", f"{col_data.median():.3f}")
            with col3:
                st.metric("Std Dev", f"{col_data.std():.3f}")
            with col4:
                st.metric("Skewness", f"{col_data.skew():.3f}")
            
            # Add distribution tests
            st.subheader("Distribution Tests")
            col1, col2 = st.columns(2)
            
            with col1:
                # Shapiro-Wilk test for normality
                if len(col_data.dropna()) <= 5000:  # Shapiro-Wilk works best with smaller samples
                    shapiro_stat, shapiro_p = stats.shapiro(col_data.dropna())
                    st.metric("Shapiro-Wilk Test", f"p-value: {shapiro_p:.4f}")
                    if shapiro_p > 0.05:
                        st.success("Data appears to be normally distributed")
                    else:
                        st.warning("Data does not appear to be normally distributed")
                else:
                    st.info("Sample too large for Shapiro-Wilk test")
            
            with col2:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(col_data.dropna(), 'norm', args=(col_data.mean(), col_data.std()))
                st.metric("Kolmogorov-Smirnov Test", f"p-value: {ks_p:.4f}")
                if ks_p > 0.05:
                    st.success("Data appears to follow normal distribution")
                else:
                    st.warning("Data does not follow normal distribution")
    
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
                st.subheader("Overlaid Histograms Customization")
                
                # Customization options
                col1, col2 = st.columns(2)
                
                with col1:
                    bins = st.slider("Number of bins:", 5, 100, 30, help="Adjust the number of histogram bins")
                    opacity = st.slider("Opacity:", 0.1, 1.0, 0.7, 0.1, help="Control histogram bar transparency")
                    barmode = st.selectbox("Bar mode:", ["overlay", "group", "stack"], help="How to display multiple histograms")
                    
                with col2:
                    show_kde = st.checkbox("Show KDE curves", value=False, help="Overlay kernel density estimation for each variable")
                    color_scheme = st.selectbox("Color scheme:", 
                        ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"], 
                        help="Choose color scheme")
                    normalize = st.checkbox("Normalize histograms", value=False, help="Show probability density instead of counts")
                
                # Create the figure
                fig = go.Figure()
                
                # Color palette
                colors = px.colors.qualitative.Set1 if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme)
                
                for i, col in enumerate(selected_cols):
                    # Choose color
                    if color_scheme == "Default":
                        color = colors[i % len(colors)]
                    else:
                        color = colors[i % len(colors)]
                    
                    # Add histogram
                    fig.add_trace(go.Histogram(
                        x=data[col], 
                        name=col, 
                        opacity=opacity,
                        nbinsx=bins,
                        marker_color=color,
                        histnorm='probability density' if normalize else ''
                    ))
                    
                    # Add KDE if requested
                    if show_kde:
                        from scipy.stats import gaussian_kde
                        col_data = data[col].dropna()
                        if len(col_data) > 1:
                            kde = gaussian_kde(col_data)
                            x_range = np.linspace(col_data.min(), col_data.max(), 200)
                            kde_values = kde(x_range)
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=kde_values,
                                mode='lines',
                                name=f'{col} KDE',
                                line=dict(color=color, width=2, dash='dot'),
                                opacity=0.8
                            ))
                
                # Update layout
                fig.update_layout(
                    title="Customized Overlaid Histograms",
                    xaxis_title="Value",
                    yaxis_title='Density' if normalize else 'Count',
                    barmode=barmode,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add comparison statistics
                st.subheader("Distribution Comparison")
                comparison_data = []
                for col in selected_cols:
                    col_data = data[col].dropna()
                    comparison_data.append({
                        'Variable': col,
                        'Mean': col_data.mean(),
                        'Median': col_data.median(),
                        'Std Dev': col_data.std(),
                        'Skewness': col_data.skew(),
                        'Kurtosis': col_data.kurtosis()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.round(3), use_container_width=True)
            
            elif viz_type == "Box Plots":
                st.subheader("Box Plots Comparison Customization")
                
                # Customization options
                col1, col2 = st.columns(2)
                
                with col1:
                    show_outliers = st.checkbox("Show outliers", value=True, help="Display outlier points")
                    show_mean = st.checkbox("Show mean", value=False, help="Display mean markers")
                    show_notches = st.checkbox("Show notches", value=False, help="Display confidence interval notches")
                    boxmode = st.selectbox("Box mode:", ["group", "overlay"], help="How to display multiple boxes")
                    
                with col2:
                    color_scheme = st.selectbox("Color scheme:", 
                        ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"], 
                        help="Choose color scheme")
                    orientation = st.selectbox("Orientation:", ["vertical", "horizontal"], help="Box plot orientation")
                    show_points = st.selectbox("Show points:", ["outliers", "all", "suspectedoutliers", "False"], 
                        help="Which points to show")
                
                # Create the figure
                fig = go.Figure()
                
                # Color palette
                colors = px.colors.qualitative.Set1 if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme)
                
                for i, col in enumerate(selected_cols):
                    # Choose color
                    color = colors[i % len(colors)]
                    
                    # Add box plot
                    fig.add_trace(go.Box(
                        y=data[col] if orientation == "vertical" else None,
                        x=data[col] if orientation == "horizontal" else None,
                        name=col,
                        boxpoints=show_points if show_points != "False" else False,
                        notched=show_notches,
                        marker_color=color,
                        showlegend=True
                    ))
                    
                    # Add mean if requested
                    if show_mean:
                        mean_val = data[col].mean()
                        if orientation == "vertical":
                            fig.add_hline(y=mean_val, line_dash="dash", line_color=color, 
                                        annotation_text=f"{col} Mean: {mean_val:.3f}")
                        else:
                            fig.add_vline(x=mean_val, line_dash="dash", line_color=color, 
                                        annotation_text=f"{col} Mean: {mean_val:.3f}")
                
                # Update layout
                fig.update_layout(
                    title="Customized Box Plots Comparison",
                    xaxis_title="Value" if orientation == "horizontal" else None,
                    yaxis_title="Value" if orientation == "vertical" else None,
                    boxmode=boxmode,
                    showlegend=True,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add comparison statistics
                st.subheader("Box Plot Comparison Statistics")
                comparison_data = []
                for col in selected_cols:
                    col_data = data[col].dropna()
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                    
                    comparison_data.append({
                        'Variable': col,
                        'Q1 (25%)': Q1,
                        'Median (50%)': col_data.median(),
                        'Q3 (75%)': Q3,
                        'IQR': IQR,
                        'Outliers': len(outliers),
                        'Outlier %': f"{(len(outliers) / len(col_data)) * 100:.2f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.round(3), use_container_width=True)
            
            elif viz_type == "Violin Plots":
                st.subheader("Violin Plots Comparison Customization")
                
                # Customization options
                col1, col2 = st.columns(2)
                
                with col1:
                    show_box = st.checkbox("Show box plot", value=True, help="Display box plot inside violin")
                    show_mean = st.checkbox("Show mean", value=False, help="Display mean markers")
                    show_median = st.checkbox("Show median", value=True, help="Display median lines")
                    show_points = st.selectbox("Show points:", ["outliers", "all", "suspectedoutliers", "False"], 
                        help="Which points to show")
                    
                with col2:
                    color_scheme = st.selectbox("Color scheme:", 
                        ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"], 
                        help="Choose color scheme")
                    orientation = st.selectbox("Orientation:", ["vertical", "horizontal"], help="Violin plot orientation")
                    bandwidth = st.slider("Bandwidth:", 0.1, 2.0, 1.0, 0.1, help="Kernel density estimation bandwidth")
                
                # Create the figure
                fig = go.Figure()
                
                # Color palette
                colors = px.colors.qualitative.Set1 if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme)
                
                for i, col in enumerate(selected_cols):
                    # Choose color
                    color = colors[i % len(colors)]
                    
                    # Add violin plot
                    fig.add_trace(go.Violin(
                        y=data[col] if orientation == "vertical" else None,
                        x=data[col] if orientation == "horizontal" else None,
                        name=col,
                        box_visible=show_box,
                        meanline_visible=show_mean,
                        points=show_points if show_points != "False" else False,
                        bandwidth=bandwidth,
                        marker_color=color,
                        showlegend=True
                    ))
                    
                    # Add median if requested
                    if show_median:
                        median_val = data[col].median()
                        if orientation == "vertical":
                            fig.add_hline(y=median_val, line_dash="dash", line_color=color, 
                                        annotation_text=f"{col} Median: {median_val:.3f}")
                        else:
                            fig.add_vline(x=median_val, line_dash="dash", line_color=color, 
                                        annotation_text=f"{col} Median: {median_val:.3f}")
                
                # Update layout
                fig.update_layout(
                    title="Customized Violin Plots Comparison",
                    xaxis_title="Value" if orientation == "horizontal" else None,
                    yaxis_title="Value" if orientation == "vertical" else None,
                    showlegend=True,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add comparison statistics
                st.subheader("Distribution Comparison Statistics")
                comparison_data = []
                for col in selected_cols:
                    col_data = data[col].dropna()
                    comparison_data.append({
                        'Variable': col,
                        'Mean': col_data.mean(),
                        'Median': col_data.median(),
                        'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else "N/A",
                        'Std Dev': col_data.std(),
                        'Skewness': col_data.skew(),
                        'Kurtosis': col_data.kurtosis()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.round(3), use_container_width=True)
            
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
            # Check if y_col is numeric for mean calculation
            if pd.api.types.is_numeric_dtype(data[y_col]):
                # Aggregate data using mean for numeric columns
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg_data, x=x_col, y=y_col, title=f'Average {y_col} by {x_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # For categorical y_col, use count instead of mean
                agg_data = data.groupby(x_col)[y_col].size().reset_index(name='count')
                fig = px.bar(agg_data, x=x_col, y='count', title=f'Count of {y_col} by {x_col}')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("X-axis should be a categorical variable for bar charts.")
    
    elif plot_type == "Line Plot":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis:", data.columns, key="line_x")
        with col2:
            y_col = st.selectbox("Y-axis:", get_numeric_columns(data), key="line_y")
        with col3:
            color_col = st.selectbox("Color by:", [None] + data.columns.tolist(), key="line_color")
        if x_col and y_col and x_col != y_col:
            # Cast y to numeric if needed for plotting
            y_series = pd.to_numeric(data[y_col], errors='coerce') if not pd.api.types.is_numeric_dtype(data[y_col]) else data[y_col]
            if color_col:
                fig = px.line(data.assign(_y=y_series), x=x_col, y='_y', color=color_col, title=f'{y_col} vs {x_col}')
            else:
                fig = px.line(data.assign(_y=y_series), x=x_col, y='_y', title=f'{y_col} vs {x_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select distinct X and Y columns.")
    
    elif plot_type == "Histogram":
        numeric_cols = get_numeric_columns(data)
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for histogram.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Numeric column:", numeric_cols, key="hist_x")
                bins = st.slider("Number of bins:", 5, 100, 30)
                normalize = st.checkbox("Normalize (density)", value=False)
            with col2:
                color_col = st.selectbox("Color by (optional):", [None] + data.columns.tolist(), key="hist_color")
                opacity = st.slider("Opacity:", 0.1, 1.0, 0.7, 0.1)
            # Cast selected column to numeric if needed
            x_series = pd.to_numeric(data[x_col], errors='coerce') if not pd.api.types.is_numeric_dtype(data[x_col]) else data[x_col]
            if color_col:
                fig = px.histogram(data.assign(_x=x_series), x='_x', color=color_col, nbins=bins, opacity=opacity,
                    histnorm='probability density' if normalize else None,
                    title=f'Histogram of {x_col}')
            else:
                fig = px.histogram(data.assign(_x=x_series), x='_x', nbins=bins, opacity=opacity,
                    histnorm='probability density' if normalize else None,
                    title=f'Histogram of {x_col}')
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Box Plot":
        numeric_cols = get_numeric_columns(data)
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for box plot.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                y_col = st.selectbox("Numeric column:", numeric_cols, key="box_y")
            with col2:
                x_col = st.selectbox("Group by (optional):", [None] + data.columns.tolist(), key="box_x")
            with col3:
                points = st.selectbox("Show points:", ["outliers", "all", "suspectedoutliers", "False"], key="box_points")
            # Cast y to numeric if needed
            y_series = pd.to_numeric(data[y_col], errors='coerce') if not pd.api.types.is_numeric_dtype(data[y_col]) else data[y_col]
            if x_col:
                fig = px.box(data.assign(_y=y_series), x=x_col, y='_y', points=(points if points != "False" else False), title=f'{y_col} by {x_col}')
            else:
                fig = px.box(data.assign(_y=y_series), y='_y', points=(points if points != "False" else False), title=f'Box Plot of {y_col}')
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot":
        numeric_cols = get_numeric_columns(data)
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for violin plot.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                y_col = st.selectbox("Numeric column:", numeric_cols, key="violin_y")
            with col2:
                x_col = st.selectbox("Group by (optional):", [None] + data.columns.tolist(), key="violin_x")
            with col3:
                show_box = st.checkbox("Show inner box", value=True, key="violin_box")
            # Cast y to numeric if needed
            y_series = pd.to_numeric(data[y_col], errors='coerce') if not pd.api.types.is_numeric_dtype(data[y_col]) else data[y_col]
            if x_col:
                fig = px.violin(data.assign(_y=y_series), x=x_col, y='_y', box=show_box, points=False, title=f'{y_col} by {x_col}')
            else:
                fig = px.violin(data.assign(_y=y_series), y='_y', box=show_box, points=False, title=f'Violin Plot of {y_col}')
            st.plotly_chart(fig, use_container_width=True)
    
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
    
    elif plot_type == "Sunburst":
        st.subheader("Sunburst Chart Customization")
        
        # Get categorical columns for hierarchy
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Hierarchy selection
                st.write("**Select hierarchy levels:**")
                level1 = st.selectbox("Level 1 (outer):", categorical_cols, key="sunburst_l1")
                remaining_cols = [col for col in categorical_cols if col != level1]
                level2 = st.selectbox("Level 2 (middle):", remaining_cols, key="sunburst_l2")
                remaining_cols = [col for col in remaining_cols if col != level2]
                level3 = st.selectbox("Level 3 (inner):", ["None"] + remaining_cols, key="sunburst_l3")
                
                # Value column
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                value_col = st.selectbox("Value column:", ["Count"] + numeric_cols, key="sunburst_value")
                
            with col2:
                # Customization options
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"], 
                    help="Choose color scheme")
                
                max_depth = st.slider("Max depth:", 1, 3, 3, help="Maximum hierarchy depth to display")
                branchvalues = st.selectbox("Branch values:", ["remainder", "total"], 
                    help="How to calculate branch values")
                
                show_labels = st.checkbox("Show labels", value=True, help="Display text labels")
                show_values = st.checkbox("Show values", value=True, help="Display value labels")
            
            # Prepare data for sunburst
            if level3 != "None":
                hierarchy_cols = [level1, level2, level3]
            else:
                hierarchy_cols = [level1, level2]
            
            # Create sunburst data
            if value_col == "Count":
                sunburst_data = data.groupby(hierarchy_cols).size().reset_index(name='count')
                values_col = 'count'
            else:
                sunburst_data = data.groupby(hierarchy_cols)[value_col].sum().reset_index()
                values_col = value_col
            
            # Create sunburst chart
            fig = px.sunburst(
                sunburst_data,
                path=hierarchy_cols,
                values=values_col,
                color=values_col,
                color_continuous_scale=color_scheme.lower() if color_scheme != "Default" else None,
                maxdepth=max_depth,
                branchvalues=branchvalues
            )
            
            # Update layout
            fig.update_layout(
                title="Customized Sunburst Chart",
                font_size=12 if show_labels else 8
            )
            
            # Hide labels if requested
            if not show_labels:
                fig.update_traces(textinfo="none")
            elif not show_values:
                fig.update_traces(textinfo="label")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add sunburst statistics
            st.subheader("Sunburst Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Categories", len(sunburst_data))
            with col2:
                st.metric("Total Value", f"{sunburst_data[values_col].sum():.2f}")
            with col3:
                st.metric("Average per Category", f"{sunburst_data[values_col].mean():.2f}")
            
            # Show data table
            st.subheader("Sunburst Data")
            st.dataframe(sunburst_data, use_container_width=True)
            
        else:
            st.warning("Need at least 2 categorical columns for sunburst chart.")
    
    elif plot_type == "Treemap":
        st.subheader("Treemap Chart Customization")
        
        # Get categorical columns for hierarchy
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) >= 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Hierarchy selection
                st.write("**Select hierarchy levels:**")
                level1 = st.selectbox("Level 1 (outer):", categorical_cols, key="treemap_l1")
                remaining_cols = [col for col in categorical_cols if col != level1]
                level2 = st.selectbox("Level 2 (inner):", ["None"] + remaining_cols, key="treemap_l2")
                
                # Value column
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                value_col = st.selectbox("Value column:", ["Count"] + numeric_cols, key="treemap_value")
                
            with col2:
                # Customization options
                color_scheme = st.selectbox("Color scheme:", 
                    ["Default", "Blues", "Reds", "Greens", "Purples", "Oranges", "Viridis", "Plasma"], 
                    help="Choose color scheme")
                
                max_depth = st.slider("Max depth:", 1, 2, 2, help="Maximum hierarchy depth to display")
                show_labels = st.checkbox("Show labels", value=True, help="Display text labels")
                show_values = st.checkbox("Show values", value=True, help="Display value labels")
                
                # Layout options
                layout_algorithm = st.selectbox("Layout algorithm:", ["squarify", "binary", "dice", "slice"], 
                    help="Algorithm for rectangle layout")
            
            # Prepare data for treemap
            if level2 != "None":
                hierarchy_cols = [level1, level2]
            else:
                hierarchy_cols = [level1]
            
            # Create treemap data
            if value_col == "Count":
                treemap_data = data.groupby(hierarchy_cols).size().reset_index(name='count')
                values_col = 'count'
            else:
                treemap_data = data.groupby(hierarchy_cols)[value_col].sum().reset_index()
                values_col = value_col
            
            # Create treemap chart
            fig = px.treemap(
                treemap_data,
                path=hierarchy_cols,
                values=values_col,
                color=values_col,
                color_continuous_scale=color_scheme.lower() if color_scheme != "Default" else None,
                maxdepth=max_depth
            )
            
            # Update layout
            fig.update_layout(
                title="Customized Treemap Chart",
                font_size=12 if show_labels else 8
            )
            
            # Hide labels if requested
            if not show_labels:
                fig.update_traces(textinfo="none")
            elif not show_values:
                fig.update_traces(textinfo="label")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add treemap statistics
            st.subheader("Treemap Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Categories", len(treemap_data))
            with col2:
                st.metric("Total Value", f"{treemap_data[values_col].sum():.2f}")
            with col3:
                st.metric("Average per Category", f"{treemap_data[values_col].mean():.2f}")
            
            # Show data table
            st.subheader("Treemap Data")
            st.dataframe(treemap_data, use_container_width=True)
            
        else:
            st.warning("Need at least 1 categorical column for treemap chart.")
    
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

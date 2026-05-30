import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_missing_data_viz(data):
    """
    Create comprehensive missing data visualizations
    """
    visualizations = {}
    
    # Missing data heatmap
    missing_data = data.isnull()
    
    if missing_data.sum().sum() > 0:
        # Plotly heatmap
        fig = px.imshow(
            missing_data.T,
            aspect="auto",
            title="Missing Data Heatmap",
            labels=dict(x="Row Index", y="Columns", color="Missing"),
            color_continuous_scale=["blue", "red"]
        )
        visualizations['heatmap'] = fig
        
        # Missing data bar chart
        missing_counts = data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            fig = px.bar(
                x=missing_counts.index,
                y=missing_counts.values,
                title="Missing Data Count by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            fig.update_xaxes(tickangle=45)
            visualizations['bar_chart'] = fig
            
            # Missing data percentage
            missing_percentages = (missing_counts / len(data) * 100).round(2)
            fig = px.bar(
                x=missing_percentages.index,
                y=missing_percentages.values,
                title="Missing Data Percentage by Column",
                labels={'x': 'Columns', 'y': 'Missing Percentage (%)'}
            )
            fig.update_xaxes(tickangle=45)
            visualizations['percentage_chart'] = fig
    
    return visualizations

def create_outlier_viz(data, column):
    """
    Create outlier visualizations for a specific column
    """
    visualizations = {}
    
    if column in data.columns and data[column].dtype in ['int64', 'float64']:
        col_data = data[column].dropna()
        
        # Box plot
        fig = px.box(y=col_data, title=f'Box Plot - {column}')
        visualizations['box_plot'] = fig
        
        # Histogram
        fig = px.histogram(col_data, title=f'Histogram - {column}', nbins=30)
        visualizations['histogram'] = fig
        
        # Scatter plot with outliers highlighted
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        is_outlier = (col_data < lower_bound) | (col_data > upper_bound)
        
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=col_data[~is_outlier].index,
            y=col_data[~is_outlier],
            mode='markers',
            name='Normal',
            marker=dict(color='blue')
        ))
        
        # Outliers
        if is_outlier.any():
            fig.add_trace(go.Scatter(
                x=col_data[is_outlier].index,
                y=col_data[is_outlier],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8)
            ))
        
        fig.update_layout(
            title=f'Outlier Detection - {column}',
            xaxis_title='Index',
            yaxis_title=column
        )
        visualizations['scatter_plot'] = fig
        
        # Z-score plot
        z_scores = np.abs(stats.zscore(col_data))
        fig = px.scatter(
            x=col_data.index,
            y=z_scores,
            title=f'Z-Score Plot - {column}',
            labels={'x': 'Index', 'y': 'Absolute Z-Score'}
        )
        fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Z-Score = 3")
        visualizations['z_score_plot'] = fig
    
    return visualizations

def create_correlation_heatmap(data):
    """
    Create correlation heatmap for numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        # Plotly heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        
        return fig
    
    return None

def create_distribution_plots(data, column):
    """
    Create distribution plots for a specific column
    """
    visualizations = {}
    
    if column in data.columns:
        col_data = data[column].dropna()
        
        if data[column].dtype in ['int64', 'float64']:
            # Histogram
            fig = px.histogram(col_data, title=f'Distribution - {column}', nbins=30)
            visualizations['histogram'] = fig
            
            # Box plot
            fig = px.box(y=col_data, title=f'Box Plot - {column}')
            visualizations['box_plot'] = fig
            
            # Violin plot
            fig = px.violin(y=col_data, title=f'Violin Plot - {column}')
            visualizations['violin_plot'] = fig
            
            # Q-Q plot
            fig = go.Figure()
            
            # Calculate theoretical quantiles
            sorted_data = np.sort(col_data)
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Data Points'
            ))
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sorted_data.min())
            max_val = max(theoretical_quantiles.max(), sorted_data.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Q-Q Plot - {column}',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles'
            )
            visualizations['qq_plot'] = fig
            
        else:
            # Categorical column
            value_counts = col_data.value_counts().head(20)
            
            # Bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Value Counts - {column}'
            )
            fig.update_xaxes(tickangle=45)
            visualizations['bar_chart'] = fig
            
            # Pie chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution - {column}'
            )
            visualizations['pie_chart'] = fig
    
    return visualizations

def create_bivariate_plot(data, x_col, y_col):
    """
    Create bivariate analysis plot
    """
    if x_col not in data.columns or y_col not in data.columns:
        return None
    
    x_is_numeric = data[x_col].dtype in ['int64', 'float64']
    y_is_numeric = data[y_col].dtype in ['int64', 'float64']
    
    if x_is_numeric and y_is_numeric:
        # Scatter plot with regression line
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col, 
            trendline="ols",
            title=f'{y_col} vs {x_col}'
        )
        return fig
    
    elif x_is_numeric and not y_is_numeric:
        # Box plot
        fig = px.box(data, x=y_col, y=x_col, title=f'{x_col} by {y_col}')
        return fig
    
    elif not x_is_numeric and y_is_numeric:
        # Box plot
        fig = px.box(data, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
        return fig
    
    else:
        # Heatmap for categorical vs categorical
        crosstab = pd.crosstab(data[x_col], data[y_col])
        fig = px.imshow(
            crosstab,
            text_auto=True,
            title=f'{y_col} vs {x_col} (Cross-tabulation)'
        )
        return fig

def create_multivariate_plot(data, columns, plot_type='scatter_matrix'):
    """
    Create multivariate analysis plots
    """
    if len(columns) < 2:
        return None
    
    if plot_type == 'scatter_matrix':
        fig = px.scatter_matrix(data[columns], title="Scatter Matrix")
        return fig
    
    elif plot_type == 'parallel_coordinates':
        # Normalize data for parallel coordinates
        normalized_data = data[columns].copy()
        for col in columns:
            if data[col].dtype in ['int64', 'float64']:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val != min_val:
                    normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        
        fig = px.parallel_coordinates(normalized_data, title="Parallel Coordinates Plot")
        return fig
    
    elif plot_type == '3d_scatter' and len(columns) >= 3:
        fig = px.scatter_3d(
            data,
            x=columns[0],
            y=columns[1],
            z=columns[2],
            title=f"3D Scatter Plot: {columns[0]} vs {columns[1]} vs {columns[2]}"
        )
        return fig
    
    return None

def create_class_imbalance_viz(data, column):
    """
    Create class imbalance visualizations
    """
    visualizations = {}
    
    if column in data.columns and data[column].dtype == 'object':
        value_counts = data[column].value_counts()
        
        # Bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Class Distribution - {column}'
        )
        fig.update_xaxes(tickangle=45)
        visualizations['bar_chart'] = fig
        
        # Pie chart
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f'Class Distribution - {column}'
        )
        visualizations['pie_chart'] = fig
        
        # Imbalance ratio visualization
        if len(value_counts) > 1:
            ratios = value_counts.values / value_counts.values[-1]  # Ratio to smallest class
            
            fig = px.bar(
                x=value_counts.index,
                y=ratios,
                title=f'Imbalance Ratios - {column}'
            )
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(title_text='Ratio to Minority Class')
            visualizations['imbalance_ratio'] = fig
    
    return visualizations

def create_skewness_viz(data, column):
    """
    Create skewness visualization with transformation suggestions
    """
    visualizations = {}
    
    if column in data.columns and data[column].dtype in ['int64', 'float64']:
        col_data = data[column].dropna()
        
        if len(col_data) > 0:
            # Original distribution
            fig = px.histogram(col_data, title=f'Original Distribution - {column}')
            visualizations['original'] = fig
            
            # Log transformation (if all values are positive)
            if col_data.min() > 0:
                log_data = np.log(col_data)
                fig = px.histogram(log_data, title=f'Log-Transformed Distribution - {column}')
                visualizations['log_transform'] = fig
            
            # Square root transformation (if all values are non-negative)
            if col_data.min() >= 0:
                sqrt_data = np.sqrt(col_data)
                fig = px.histogram(sqrt_data, title=f'Square Root-Transformed Distribution - {column}')
                visualizations['sqrt_transform'] = fig
            
            # Box-Cox transformation
            try:
                if col_data.min() > 0:
                    boxcox_data, lambda_val = stats.boxcox(col_data)
                    fig = px.histogram(
                        boxcox_data, 
                        title=f'Box-Cox Transformed Distribution - {column} (λ={lambda_val:.3f})'
                    )
                    visualizations['boxcox_transform'] = fig
            except:
                pass
    
    return visualizations

def create_data_overview_dashboard(data):
    """
    Create comprehensive data overview dashboard
    """
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Types', 'Missing Data', 'Numerical Summary', 'Categorical Summary'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "table"}, {"type": "table"}]]
    )
    
    # Data types pie chart
    dtype_counts = data.dtypes.value_counts()
    fig.add_trace(
        go.Pie(
            labels=dtype_counts.index.astype(str),
            values=dtype_counts.values,
            name="Data Types"
        ),
        row=1, col=1
    )
    
    # Missing data bar chart
    missing_counts = data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    
    if len(missing_counts) > 0:
        fig.add_trace(
            go.Bar(
                x=missing_counts.index,
                y=missing_counts.values,
                name="Missing Data"
            ),
            row=1, col=2
        )
    
    # Numerical summary table
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_summary = data[numeric_cols].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic'] + numeric_summary.columns.tolist()),
                cells=dict(values=[numeric_summary.index.tolist()] + 
                          [numeric_summary[col].round(3).tolist() for col in numeric_summary.columns])
            ),
            row=2, col=1
        )
    
    # Categorical summary table
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_data = []
        for col in cat_cols[:5]:  # Limit to first 5 columns
            cat_data.append([
                col,
                data[col].nunique(),
                data[col].mode().iloc[0] if not data[col].mode().empty else 'N/A'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Column', 'Unique Values', 'Most Frequent']),
                cells=dict(values=list(zip(*cat_data)) if cat_data else [[], [], []])
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Data Overview Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig

def create_quality_score_gauge(completeness, uniqueness, consistency, validity):
    """
    Create quality score gauge chart
    """
    # Calculate overall score
    overall_score = (completeness * 0.3 + uniqueness * 0.3 + consistency * 0.2 + validity * 0.2)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Quality Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    return fig

def create_comparison_chart(original_data, cleaned_data, metric_name):
    """
    Create before/after comparison chart
    """
    if metric_name == 'missing_values':
        original_missing = original_data.isnull().sum().sum()
        cleaned_missing = cleaned_data.isnull().sum().sum()
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=['Missing Values'], y=[original_missing]),
            go.Bar(name='Cleaned', x=['Missing Values'], y=[cleaned_missing])
        ])
        
        fig.update_layout(
            title='Missing Values: Before vs After',
            barmode='group'
        )
        
        return fig
    
    elif metric_name == 'duplicates':
        original_duplicates = original_data.duplicated().sum()
        cleaned_duplicates = cleaned_data.duplicated().sum()
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=['Duplicate Rows'], y=[original_duplicates]),
            go.Bar(name='Cleaned', x=['Duplicate Rows'], y=[cleaned_duplicates])
        ])
        
        fig.update_layout(
            title='Duplicate Rows: Before vs After',
            barmode='group'
        )
        
        return fig
    
    elif metric_name == 'data_shape':
        fig = go.Figure(data=[
            go.Bar(name='Original', x=['Rows', 'Columns'], y=[original_data.shape[0], original_data.shape[1]]),
            go.Bar(name='Cleaned', x=['Rows', 'Columns'], y=[cleaned_data.shape[0], cleaned_data.shape[1]])
        ])
        
        fig.update_layout(
            title='Data Shape: Before vs After',
            barmode='group'
        )
        
        return fig
    
    return None

def create_feature_importance_plot(data, target_column):
    """
    Create feature importance plot for categorical target
    """
    if target_column not in data.columns:
        return None
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None
    
    # Calculate correlation with target (if target is numeric)
    if data[target_column].dtype in ['int64', 'float64']:
        correlations = data[numeric_cols].corrwith(data[target_column]).abs().sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.index,
            y=correlations.values,
            title=f'Feature Correlation with {target_column}'
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    return None


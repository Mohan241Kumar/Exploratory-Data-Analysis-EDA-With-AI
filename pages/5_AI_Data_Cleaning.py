import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
from utils.data_quality import detect_all_issues
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Data Cleaning",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Data Cleaning")
st.markdown("Intelligent data cleaning suggestions and automated fixes using AI")

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

def get_ai_cleaning_suggestions(data_info, issues):
    """Get AI suggestions for data cleaning"""
    
    prompt = f"""
    You are a data science expert. Analyze this dataset and provide specific cleaning recommendations.
    
    Dataset Info:
    - Shape: {data_info['shape']}
    - Columns: {data_info['columns']}
    - Data Types: {data_info['dtypes']}
    
    Detected Issues:
    {json.dumps(issues, indent=2, default=str)}
    
    Please provide:
    1. Priority ranking of issues (1-10, where 10 is most critical)
    2. Specific cleaning strategies for each issue
    3. Potential risks of each cleaning approach
    4. Recommended order of operations
    5. Expected impact on data quality
    
    Respond in JSON format with this structure:
    {{
        "overall_assessment": "Brief summary of data quality",
        "priority_issues": [
            {{
                "issue": "issue_name",
                "priority": 1-10,
                "description": "What this issue means",
                "strategy": "Recommended cleaning approach",
                "risks": "Potential risks",
                "impact": "Expected improvement"
            }}
        ],
        "cleaning_order": ["step1", "step2", "step3"],
        "expected_quality_improvement": "percentage or description"
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
        st.error(f"Error getting AI suggestions: {str(e)}")
        return None

def apply_ai_cleaning(data, cleaning_plan):
    """Apply AI-recommended cleaning operations"""
    cleaned_data = data.copy()
    cleaning_log = []
    
    for step in cleaning_plan.get('priority_issues', []):
        issue = step['issue']
        strategy = step['strategy']
        
        if 'missing' in issue.lower():
            # Handle missing data
            for col in cleaned_data.columns:
                if cleaned_data[col].isnull().sum() > 0:
                    if cleaned_data[col].dtype in ['int64', 'float64']:
                        if 'median' in strategy.lower():
                            median_val = cleaned_data[col].median()
                            cleaned_data[col].fillna(median_val, inplace=True)
                            cleaning_log.append(f"Filled missing values in '{col}' with median: {median_val}")
                        elif 'mean' in strategy.lower():
                            mean_val = cleaned_data[col].mean()
                            cleaned_data[col].fillna(mean_val, inplace=True)
                            cleaning_log.append(f"Filled missing values in '{col}' with mean: {mean_val}")
                    else:
                        mode_val = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown'
                        cleaned_data[col].fillna(mode_val, inplace=True)
                        cleaning_log.append(f"Filled missing values in '{col}' with mode: '{mode_val}'")
        
        elif 'duplicate' in issue.lower():
            # Remove duplicates
            duplicates_before = cleaned_data.duplicated().sum()
            cleaned_data.drop_duplicates(inplace=True)
            cleaning_log.append(f"Removed {duplicates_before} duplicate rows")
        
        elif 'outlier' in issue.lower():
            # Handle outliers
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
                if outliers_before > 0:
                    if 'cap' in strategy.lower() or 'clip' in strategy.lower():
                        cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                        cleaning_log.append(f"Capped {outliers_before} outliers in '{col}'")
        
        elif 'type' in issue.lower():
            # Fix data types
            for col in cleaned_data.columns:
                if cleaned_data[col].dtype == 'object':
                    # Try numeric conversion
                    try:
                        numeric_series = pd.to_numeric(cleaned_data[col], errors='coerce')
                        if numeric_series.notna().sum() > len(cleaned_data) * 0.8:
                            cleaned_data[col] = numeric_series
                            cleaning_log.append(f"Converted '{col}' to numeric type")
                    except:
                        pass
    
    return cleaned_data, cleaning_log

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 AI Analysis")
    
    if st.button("🚀 Get AI Cleaning Recommendations", type="primary"):
        with st.spinner("AI is analyzing your data..."):
            # Get data info
            data_info = {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Detect issues
            issues = detect_all_issues(data)
            
            # Get AI suggestions
            ai_suggestions = get_ai_cleaning_suggestions(data_info, issues)
            
            if ai_suggestions:
                st.session_state.ai_suggestions = ai_suggestions
                st.success("AI analysis completed!")

with col2:
    st.header("📊 Quick Stats")
    st.metric("Rows", f"{data.shape[0]:,}")
    st.metric("Columns", f"{data.shape[1]:,}")
    st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    st.metric("Duplicates", f"{data.duplicated().sum():,}")

# Display AI recommendations
if 'ai_suggestions' in st.session_state:
    suggestions = st.session_state.ai_suggestions
    
    st.header("🧠 AI Recommendations")
    
    # Overall assessment
    st.subheader("📋 Overall Assessment")
    st.info(suggestions.get('overall_assessment', 'No assessment available'))
    
    # Priority issues
    st.subheader("⚡ Priority Issues")
    
    priority_issues = suggestions.get('priority_issues', [])
    if priority_issues:
        for i, issue in enumerate(priority_issues):
            with st.expander(f"#{i+1} - {issue.get('issue', 'Unknown').title()} (Priority: {issue.get('priority', 0)}/10)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Description:**")
                    st.write(issue.get('description', 'No description'))
                    
                    st.write("**Strategy:**")
                    st.write(issue.get('strategy', 'No strategy'))
                
                with col2:
                    st.write("**Risks:**")
                    st.warning(issue.get('risks', 'No risks identified'))
                    
                    st.write("**Expected Impact:**")
                    st.success(issue.get('impact', 'No impact specified'))
    
    # Cleaning order
    st.subheader("📋 Recommended Cleaning Order")
    cleaning_order = suggestions.get('cleaning_order', [])
    if cleaning_order:
        for i, step in enumerate(cleaning_order, 1):
            st.write(f"{i}. {step}")
    
    # Expected improvement
    st.subheader("📈 Expected Quality Improvement")
    improvement = suggestions.get('expected_quality_improvement', 'Not specified')
    st.metric("Quality Improvement", improvement)
    
    # Apply AI cleaning
    st.header("🛠️ Apply AI Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Apply All AI Recommendations", type="primary"):
            with st.spinner("Applying AI cleaning recommendations..."):
                cleaned_data, cleaning_log = apply_ai_cleaning(data, suggestions)
                
                st.session_state.ai_cleaned_data = cleaned_data
                st.session_state.ai_cleaning_log = cleaning_log
                
                st.success(f"Applied {len(cleaning_log)} cleaning operations!")
    
    with col2:
        if st.button("📝 View Detailed Plan"):
            st.json(suggestions)

# Display cleaning results
if 'ai_cleaned_data' in st.session_state and 'ai_cleaning_log' in st.session_state:
    st.header("✅ AI Cleaning Results")
    
    # Cleaning log
    st.subheader("📋 Operations Applied")
    for i, operation in enumerate(st.session_state.ai_cleaning_log, 1):
        st.write(f"{i}. {operation}")
    
    # Before/After comparison
    st.subheader("⚖️ Before vs After Comparison")
    
    original_data = st.session_state.data
    cleaned_data = st.session_state.ai_cleaned_data
    
    comparison_metrics = pd.DataFrame({
        'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicates', 'Memory (MB)'],
        'Before': [
            f"{original_data.shape[0]:,}",
            f"{original_data.shape[1]:,}",
            f"{original_data.isnull().sum().sum():,}",
            f"{original_data.duplicated().sum():,}",
            f"{original_data.memory_usage(deep=True).sum() / 1024**2:.2f}"
        ],
        'After': [
            f"{cleaned_data.shape[0]:,}",
            f"{cleaned_data.shape[1]:,}",
            f"{cleaned_data.isnull().sum().sum():,}",
            f"{cleaned_data.duplicated().sum():,}",
            f"{cleaned_data.memory_usage(deep=True).sum() / 1024**2:.2f}"
        ]
    })
    
    st.dataframe(comparison_metrics, use_container_width=True)
    
    # Quality improvement visualization
    st.subheader("📊 Quality Improvement Visualization")
    
    # Create improvement metrics
    before_missing = original_data.isnull().sum().sum()
    after_missing = cleaned_data.isnull().sum().sum()
    before_duplicates = original_data.duplicated().sum()
    after_duplicates = cleaned_data.duplicated().sum()
    
    fig = go.Figure(data=[
        go.Bar(name='Before', x=['Missing Values', 'Duplicates'], y=[before_missing, before_duplicates]),
        go.Bar(name='After', x=['Missing Values', 'Duplicates'], y=[after_missing, after_duplicates])
    ])
    
    fig.update_layout(
        title='Data Quality Improvement',
        barmode='group',
        yaxis_title='Count'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.subheader("👀 Cleaned Data Preview")
    st.dataframe(cleaned_data.head(10), use_container_width=True)
    
    # Download cleaned data
    st.subheader("📥 Download Cleaned Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = cleaned_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="ai_cleaned_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create cleaning report
        cleaning_report = {
            'original_shape': original_data.shape,
            'cleaned_shape': cleaned_data.shape,
            'operations_applied': st.session_state.ai_cleaning_log,
            'ai_recommendations': st.session_state.ai_suggestions,
            'quality_improvement': {
                'missing_values_reduced': before_missing - after_missing,
                'duplicates_removed': before_duplicates - after_duplicates
            }
        }
        
        st.download_button(
            label="📊 Download Cleaning Report",
            data=json.dumps(cleaning_report, indent=2, default=str),
            file_name="ai_cleaning_report.json",
            mime="application/json"
        )

# AI Insights section
st.header("💡 AI Data Insights")

if st.button("🔮 Generate AI Insights"):
    with st.spinner("AI is generating insights..."):
        # Prepare data summary for AI
        data_summary = {
            'shape': data.shape,
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
            'missing_summary': data.isnull().sum().to_dict(),
            'basic_stats': data.describe().to_dict()
        }
        
        insight_prompt = f"""
        Analyze this dataset and provide actionable business insights:
        
        {json.dumps(data_summary, indent=2, default=str)}
        
        Please provide:
        1. Key patterns and trends you notice
        2. Potential business insights
        3. Recommendations for further analysis
        4. Data quality observations
        5. Suggested visualizations
        
        Respond in JSON format:
        {{
            "key_patterns": ["pattern1", "pattern2"],
            "business_insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "quality_observations": ["obs1", "obs2"],
            "suggested_visualizations": ["viz1", "viz2"]
        }}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[{"role": "user", "content": insight_prompt}],
                response_format={"type": "json_object"}
            )
            
            insights = json.loads(response.choices[0].message.content)
            
            # Display insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 Key Patterns")
                for pattern in insights.get('key_patterns', []):
                    st.write(f"• {pattern}")
                
                st.subheader("💼 Business Insights")
                for insight in insights.get('business_insights', []):
                    st.write(f"• {insight}")
            
            with col2:
                st.subheader("📋 Recommendations")
                for rec in insights.get('recommendations', []):
                    st.write(f"• {rec}")
                
                st.subheader("🎨 Suggested Visualizations")
                for viz in insights.get('suggested_visualizations', []):
                    st.write(f"• {viz}")
            
            st.subheader("🔍 Quality Observations")
            for obs in insights.get('quality_observations', []):
                st.info(f"💡 {obs}")
                
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")

st.markdown("---")
st.markdown("**AI Data Cleaning** - Intelligent automation for data preprocessing")
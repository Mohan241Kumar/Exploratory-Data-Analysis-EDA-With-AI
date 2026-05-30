import streamlit as st
import pandas as pd
import numpy as np
from utils.ml_recommendations import (
    get_ml_recommendations, 
    analyze_target_variable, 
    suggest_algorithms,
    evaluate_data_readiness,
    generate_model_code
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML Recommendations", page_icon="🧠", layout="wide")

st.title("🧠 Machine Learning Recommendations")

if st.session_state.get('data') is None:
    st.warning("⚠️ Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

st.markdown("""
Get intelligent machine learning model recommendations based on your dataset characteristics.
Our AI analyzes your data to suggest the most suitable algorithms, preprocessing steps, and evaluation strategies.
""")

# ML Task Configuration
st.subheader("🎯 Define Your ML Task")

col1, col2 = st.columns(2)

with col1:
    task_type = st.selectbox(
        "Select ML Task Type",
        ["Auto-Detect", "Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series"],
        help="Choose your machine learning objective"
    )

with col2:
    target_column = st.selectbox(
        "Select Target Variable (if applicable)",
        options=["None"] + list(data.columns),
        help="Choose the column you want to predict (for supervised learning)"
    )

# Auto-detect task type if selected
if task_type == "Auto-Detect" and target_column != "None":
    with st.spinner("🔍 Auto-detecting ML task type..."):
        detected_task = analyze_target_variable(data, target_column)
        st.success(f"✅ Detected task type: **{detected_task}**")
        task_type = detected_task

# Generate ML Recommendations
if st.button("🚀 Generate ML Recommendations", type="primary"):
    
    with st.spinner("🧠 Analyzing dataset and generating recommendations..."):
        try:
            # Get comprehensive recommendations
            recommendations = get_ml_recommendations(
                data, 
                task_type=task_type, 
                target_column=target_column if target_column != "None" else None
            )
            
            # Store in session state
            st.session_state.ml_recommendations = recommendations
            
            st.success("✅ ML recommendations generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            st.stop()

# Display recommendations if available
if st.session_state.get('ml_recommendations'):
    recommendations = st.session_state.ml_recommendations
    
    # Task Summary
    st.subheader("📋 Task Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Task Type", recommendations.get('task_type', 'Unknown'))
    with col2:
        st.metric("Dataset Size", f"{len(data):,} rows")
    with col3:
        st.metric("Features", len(data.columns))
    with col4:
        readiness_score = recommendations.get('data_readiness_score', 0)
        st.metric("Data Readiness", f"{readiness_score:.1f}/10")
    
    # Data Readiness Assessment
    st.subheader("🏥 Data Readiness Assessment")
    
    readiness = recommendations.get('data_readiness', {})
    
    if readiness:
        readiness_df = pd.DataFrame([
            {'Aspect': k, 'Score': v, 'Status': '✅ Good' if v >= 7 else '⚠️ Needs Work' if v >= 4 else '❌ Poor'}
            for k, v in readiness.items()
        ])
        
        fig = px.bar(
            readiness_df,
            x='Aspect',
            y='Score',
            color='Score',
            title="Data Readiness by Aspect",
            color_continuous_scale='RdYlGn',
            range_color=[0, 10]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)
    
    # Algorithm Recommendations
    st.subheader("🎯 Algorithm Recommendations")
    
    algorithms = recommendations.get('recommended_algorithms', [])
    
    if algorithms:
        for i, algo in enumerate(algorithms):
            with st.expander(f"🥇 Rank {i+1}: {algo.get('name', 'Unknown Algorithm')} (Score: {algo.get('score', 0):.2f})"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Algorithm Details:**")
                    st.write(f"- **Type:** {algo.get('type', 'Unknown')}")
                    st.write(f"- **Complexity:** {algo.get('complexity', 'Unknown')}")
                    st.write(f"- **Training Time:** {algo.get('training_time', 'Unknown')}")
                    st.write(f"- **Interpretability:** {algo.get('interpretability', 'Unknown')}")
                
                with col2:
                    st.write("**Why This Algorithm:**")
                    for reason in algo.get('reasons', []):
                        st.write(f"• {reason}")
                
                st.write("**Description:**")
                st.write(algo.get('description', 'No description available.'))
                
                # Show hyperparameters
                hyperparams = algo.get('hyperparameters', {})
                if hyperparams:
                    st.write("**Suggested Hyperparameters:**")
                    st.json(hyperparams)
                
                # Show code example
                if st.button(f"📝 Generate Code for {algo.get('name')}", key=f"code_{i}"):
                    target_for_code = None if target_column == "None" else target_column
                    code = generate_model_code(algo, task_type, target_for_code)
                    st.code(code, language='python')
    
    # Feature Engineering Suggestions
    st.subheader("⚙️ Feature Engineering Suggestions")
    
    feature_suggestions = recommendations.get('feature_engineering', [])
    
    if feature_suggestions:
        for suggestion in feature_suggestions:
            st.write(f"🔧 **{suggestion.get('technique', 'Unknown')}**")
            st.write(f"   *Reason:* {suggestion.get('reason', 'No reason provided')}")
            st.write(f"   *Impact:* {suggestion.get('impact', 'Unknown impact')}")
            
            if suggestion.get('columns'):
                st.write(f"   *Applies to:* {', '.join(suggestion.get('columns', []))}")
            
            st.write("---")
    else:
        st.info("💡 Your data appears to be ready for modeling with minimal feature engineering.")
    
    # Model Evaluation Strategy
    st.subheader("📊 Model Evaluation Strategy")
    
    evaluation = recommendations.get('evaluation_strategy', {})
    
    if evaluation:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recommended Metrics:**")
            metrics = evaluation.get('metrics', [])
            for metric in metrics:
                st.write(f"• {metric}")
        
        with col2:
            st.write("**Validation Strategy:**")
            st.write(f"• **Method:** {evaluation.get('validation_method', 'Unknown')}")
            st.write(f"• **Splits:** {evaluation.get('cv_folds', 'Unknown')}")
            st.write(f"• **Test Size:** {evaluation.get('test_size', 'Unknown')}")
    
    # Preprocessing Recommendations
    st.subheader("🔧 Preprocessing Recommendations")
    
    preprocessing = recommendations.get('preprocessing_steps', [])
    
    if preprocessing:
        for step in preprocessing:
            priority = step.get('priority', 'Medium')
            priority_icons = {'High': '🔥', 'Medium': '⚡', 'Low': '💡'}
            priority_icon = priority_icons.get(priority, '📌')
            
            st.write(f"{priority_icon} **{step.get('step', 'Unknown Step')}** ({priority} Priority)")
            st.write(f"   *Description:* {step.get('description', 'No description')}")
            
            if step.get('columns'):
                st.write(f"   *Affects:* {', '.join(step.get('columns', []))}")
    
    # Advanced Recommendations
    st.subheader("🚀 Advanced Recommendations")
    
    with st.expander("💡 Performance Optimization Tips"):
        perf_tips = recommendations.get('performance_tips', [])
        if perf_tips:
            for tip in perf_tips:
                st.write(f"• {tip}")
        else:
            st.write("• Consider feature selection to reduce dimensionality")
            st.write("• Use cross-validation for robust model evaluation")
            st.write("• Monitor for overfitting with validation curves")
            st.write("• Consider ensemble methods for improved performance")
    
    with st.expander("⚠️ Potential Challenges"):
        challenges = recommendations.get('challenges', [])
        if challenges:
            for challenge in challenges:
                st.write(f"• {challenge}")
        else:
            st.write("• Watch for data leakage in feature engineering")
            st.write("• Ensure representative train/test splits")
            st.write("• Monitor model performance on unseen data")
    
    leakage_warnings = recommendations.get('leakage_warnings', [])
    if leakage_warnings:
        with st.expander("🧯 Potential Leakage Warnings"):
            for warn in leakage_warnings:
                st.write(f"• {warn}")
    
    with st.expander("📚 Learning Resources"):
        st.write("""
        **Recommended Learning Resources:**
        
        • **Scikit-learn Documentation**: Comprehensive guides for each algorithm
        • **Kaggle Learn**: Free micro-courses on machine learning
        • **Fast.ai**: Practical deep learning courses
        • **Papers with Code**: Latest research and implementations
        • **Google's ML Crash Course**: Fundamentals of machine learning
        """)

# Interactive Model Comparison
st.subheader("⚖️ Interactive Model Comparison")

if st.session_state.get('ml_recommendations'):
    algorithms = st.session_state.ml_recommendations.get('recommended_algorithms', [])
    
    if len(algorithms) >= 2:
        st.write("Compare different algorithms across various criteria:")
        
        # Create comparison dataframe
        comparison_data = []
        for algo in algorithms[:5]:  # Top 5 algorithms
            comparison_data.append({
                'Algorithm': algo.get('name', 'Unknown'),
                'Score': algo.get('score', 0),
                'Complexity': algo.get('complexity_score', 5),
                'Speed': algo.get('speed_score', 5),
                'Interpretability': algo.get('interpretability_score', 5),
                'Accuracy Potential': algo.get('accuracy_potential', 5)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Radar chart comparison
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Score'], row['Complexity'], row['Speed'], 
                   row['Interpretability'], row['Accuracy Potential']],
                theta=['Overall Score', 'Complexity', 'Speed', 'Interpretability', 'Accuracy Potential'],
                fill='toself',
                name=row['Algorithm']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Algorithm Comparison Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Quick Baseline Runner (supervised tasks)
if st.session_state.get('ml_recommendations'):
    recommendations = st.session_state.ml_recommendations
    top_algo = (recommendations.get('recommended_algorithms') or [])[:1]
    if top_algo and recommendations.get('task_type', '').lower() in ['classification', 'binary classification', 'regression'] and target_column != 'None':
        if st.button("⚡ Run Quick Baseline with Top Algorithm"):
            algo = top_algo[0]
            code = generate_model_code(algo, recommendations.get('task_type', ''), target_column)
            st.code(code, language='python')

# Export Recommendations
if st.session_state.get('ml_recommendations'):
    st.subheader("📥 Export Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download Report as JSON"):
            import json
            report_json = json.dumps(st.session_state.ml_recommendations, indent=2)
            st.download_button(
                label="💾 Download JSON Report",
                data=report_json,
                file_name="ml_recommendations.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📝 Generate Python Script"):
            # Ensure we have local references even if blocks above didn't run this cycle
            recommendations = st.session_state.ml_recommendations
            preprocessing = recommendations.get('preprocessing_steps', [])
            feature_suggestions = recommendations.get('feature_engineering', [])
            # Generate a complete ML pipeline script
            script = f"""
# Generated ML Pipeline Script
# Task Type: {recommendations.get('task_type', 'Unknown')}
# Target Variable: {target_column if target_column != 'None' else 'None'}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load your data
# data = pd.read_csv('your_dataset.csv')

# Preprocessing steps
{chr(10).join([f"# {step.get('step', 'Unknown')}: {step.get('description', '')}" 
               for step in preprocessing])}

# Feature engineering
{chr(10).join([f"# {sugg.get('technique', 'Unknown')}: {sugg.get('reason', '')}" 
               for sugg in feature_suggestions])}

# Model training and evaluation
# Add your model training code here based on the recommendations above
"""
            
            st.download_button(
                label="💾 Download Python Script",
                data=script,
                file_name="ml_pipeline.py",
                mime="text/x-python"
            )

# Tips and Best Practices
st.subheader("💡 ML Best Practices")

st.info("""
**Key Tips for Success:**

🎯 **Start Simple**: Begin with simpler algorithms before moving to complex ones
📊 **Validate Properly**: Use appropriate cross-validation strategies
🔍 **Feature Selection**: Remove irrelevant features to improve performance
⚖️ **Handle Imbalance**: Address class imbalance in classification tasks
📈 **Monitor Performance**: Track metrics that matter for your use case
🧪 **Experiment**: Try different algorithms and hyperparameters
📝 **Document Everything**: Keep track of experiments and results
""")

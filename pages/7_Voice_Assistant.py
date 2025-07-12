import streamlit as st
import pandas as pd
import numpy as np
import speech_recognition as sr
from openai import OpenAI
import json
import os
import io
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice Assistant & Question Answering")
st.markdown("Ask questions about your data using voice or text")

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

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize speech recognizer
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = sr.Recognizer()

def get_data_context():
    """Get comprehensive data context for AI"""
    context = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'summary_stats': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
        'sample_data': data.head(3).to_dict()
    }
    return context

def answer_data_question(question, data_context):
    """Get AI answer for data-related questions"""
    
    context_str = f"""
    Dataset Information:
    - Shape: {data_context['shape'][0]} rows, {data_context['shape'][1]} columns
    - Columns: {', '.join(data_context['columns'])}
    - Numeric columns: {', '.join(data_context['numeric_columns'])}
    - Categorical columns: {', '.join(data_context['categorical_columns'])}
    - Missing values: {data_context['missing_values']}
    - Sample data: {data_context['sample_data']}
    """
    
    prompt = f"""
    You are a data analysis expert. Answer the user's question about their dataset.
    
    {context_str}
    
    User Question: {question}
    
    Provide a clear, concise answer. If the question requires specific calculations or analysis, 
    provide the exact values from the data. If suggesting visualizations or further analysis, 
    be specific about what columns to use.
    
    Format your response as JSON:
    {{
        "answer": "Your detailed answer here",
        "suggestions": ["Suggestion 1", "Suggestion 2"],
        "visualization_recommendation": {{
            "type": "bar/line/scatter/histogram/etc or null",
            "x_column": "column name or null",
            "y_column": "column name or null",
            "description": "What this chart would show"
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
        st.error(f"Error getting AI answer: {str(e)}")
        return {"answer": "Sorry, I couldn't process your question.", "suggestions": [], "visualization_recommendation": None}

def create_recommended_visualization(viz_config):
    """Create visualization based on AI recommendation"""
    if not viz_config or viz_config.get('type') is None:
        return None
    
    chart_type = viz_config.get('type')
    x_col = viz_config.get('x_column')
    y_col = viz_config.get('y_column')
    
    try:
        if chart_type == 'bar' and x_col and x_col in data.columns:
            if data[x_col].dtype == 'object':
                value_counts = data[x_col].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f'Distribution of {x_col}')
                fig.update_xaxis(title=x_col)
                fig.update_yaxis(title='Count')
            else:
                fig = px.histogram(data, x=x_col, title=f'Distribution of {x_col}')
            return fig
            
        elif chart_type == 'line' and x_col and y_col and x_col in data.columns and y_col in data.columns:
            fig = px.line(data, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            return fig
            
        elif chart_type == 'scatter' and x_col and y_col and x_col in data.columns and y_col in data.columns:
            fig = px.scatter(data, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
            return fig
            
        elif chart_type == 'histogram' and x_col and x_col in data.columns:
            fig = px.histogram(data, x=x_col, title=f'Distribution of {x_col}')
            return fig
            
        elif chart_type == 'box' and y_col and y_col in data.columns:
            fig = px.box(data, y=y_col, title=f'Box Plot of {y_col}')
            return fig
            
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
    
    return None

# Sidebar for voice settings
st.sidebar.header("🎤 Voice Settings")

# Voice input section
st.header("🎙️ Voice Input")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎤 Start Recording", type="primary", use_container_width=True):
        try:
            with sr.Microphone() as source:
                st.info("🎧 Listening... Speak your question about the data")
                
                # Adjust for ambient noise
                st.session_state.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Record audio with timeout
                audio = st.session_state.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                
                st.info("🔄 Processing speech...")
                
                # Convert speech to text
                try:
                    question = st.session_state.recognizer.recognize_google(audio)
                    st.success(f"🎯 Recognized: {question}")
                    st.session_state.current_question = question
                    
                except sr.UnknownValueError:
                    st.error("Could not understand the audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"Error with speech recognition service: {e}")
                    
        except Exception as e:
            st.error(f"Error accessing microphone: {str(e)}")
            st.info("💡 Make sure your browser allows microphone access and you have a working microphone.")

with col2:
    st.info("**Voice Commands Examples:**\n- What are the main columns?\n- Show me missing values\n- What's the average of [column]?\n- Are there any outliers?\n- Create a chart of [column]")

# Text input as alternative
st.header("💬 Text Input")

question_input = st.text_input(
    "Type your question about the data:",
    value=st.session_state.get('current_question', ''),
    placeholder="e.g., What are the main insights from this dataset?"
)

if st.button("📝 Ask Question", type="primary"):
    if question_input:
        st.session_state.current_question = question_input

# Process question if available
if 'current_question' in st.session_state and st.session_state.current_question:
    question = st.session_state.current_question
    
    with st.spinner("🤔 Analyzing your question..."):
        # Get data context
        data_context = get_data_context()
        
        # Get AI answer
        ai_response = answer_data_question(question, data_context)
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            'timestamp': datetime.now(),
            'question': question,
            'response': ai_response
        })
        
        # Display response
        st.subheader("🤖 AI Response")
        
        # Main answer
        st.write("**Answer:**")
        st.info(ai_response.get('answer', 'No answer available.'))
        
        # Suggestions
        if ai_response.get('suggestions'):
            st.write("**Suggestions for further analysis:**")
            for suggestion in ai_response['suggestions']:
                st.write(f"• {suggestion}")
        
        # Recommended visualization
        viz_config = ai_response.get('visualization_recommendation')
        if viz_config and viz_config.get('type'):
            st.write("**Recommended Visualization:**")
            st.write(f"📊 {viz_config.get('description', 'Visualization')}")
            
            fig = create_recommended_visualization(viz_config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not create the recommended visualization.")
    
    # Clear current question
    if 'current_question' in st.session_state:
        del st.session_state.current_question

# Quick question buttons
st.header("⚡ Quick Questions")

quick_questions = [
    "What are the main columns in this dataset?",
    "Are there any missing values?",
    "What are the key statistics?",
    "Show me data quality issues",
    "What patterns do you see?",
    "Suggest visualizations for this data"
]

cols = st.columns(3)
for i, q in enumerate(quick_questions):
    with cols[i % 3]:
        if st.button(q, key=f"quick_{i}"):
            st.session_state.current_question = q
            st.rerun()

# Conversation history
if st.session_state.conversation_history:
    st.header("💬 Conversation History")
    
    # Show recent conversations
    for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
        with st.expander(f"Q: {conv['question'][:50]}..." if len(conv['question']) > 50 else f"Q: {conv['question']}"):
            st.write(f"**Asked:** {conv['timestamp'].strftime('%H:%M:%S')}")
            st.write(f"**Question:** {conv['question']}")
            st.write(f"**Answer:** {conv['response']['answer']}")
            
            if conv['response'].get('suggestions'):
                st.write("**Suggestions:**")
                for suggestion in conv['response']['suggestions']:
                    st.write(f"• {suggestion}")

# Data insights panel
st.header("📊 Quick Data Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", f"{data.shape[0]:,}")
    st.metric("Total Columns", f"{data.shape[1]:,}")

with col2:
    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    st.metric("Missing Data", f"{missing_pct:.1f}%")
    st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))

with col3:
    st.metric("Categorical Columns", len(data.select_dtypes(include=['object']).columns))
    st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Export conversation
if st.session_state.conversation_history:
    st.header("📥 Export Conversation")
    
    if st.button("💾 Download Conversation History"):
        conversation_data = []
        for conv in st.session_state.conversation_history:
            conversation_data.append({
                'timestamp': conv['timestamp'].isoformat(),
                'question': conv['question'],
                'answer': conv['response']['answer'],
                'suggestions': conv['response'].get('suggestions', [])
            })
        
        conversation_json = json.dumps(conversation_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=conversation_json,
            file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Clear conversation button
if st.session_state.conversation_history:
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")
        st.rerun()

st.markdown("---")
st.markdown("**Voice Assistant** - Interact with your data using natural language")
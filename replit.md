# Advanced AI-Powered EDA Tool

## Overview

This is a comprehensive Exploratory Data Analysis (EDA) tool built with Streamlit that provides automated data profiling, quality assessment, interactive analysis, and AI-powered insights. The application combines traditional EDA capabilities with cutting-edge AI features to deliver intelligent data cleaning recommendations and automated dashboard generation similar to Power BI. Users can understand their datasets through various visualization techniques, automated reports, and AI-driven insights while identifying and addressing common data quality issues.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit-based web interface with multiple pages
- **Backend**: Python-based data processing and analysis engine
- **Utilities**: Modular helper functions for data quality, EDA operations, and visualizations
- **Session Management**: Streamlit session state for data persistence across pages

## Key Components

### 1. Main Application (`app.py`)
- Entry point and data upload functionality
- Session state management for data persistence
- Initial data quality assessment integration
- Navigation hub for other analysis pages

### 2. Multi-Page Structure
- **Automated Profiling**: Integration with pandas-profiling, Sweetviz, and AutoViz
- **Data Quality Issues**: Comprehensive analysis of 10 common data problems
- **Interactive Analysis**: Dynamic visualizations and custom analysis tools
- **Data Cleaning**: Preprocessing operations with downloadable cleaned datasets
- **AI Data Cleaning**: AI-powered intelligent cleaning recommendations and automation
- **Auto Dashboard**: Power BI-style automated dashboard generation with AI insights

### 3. Utility Modules
- **data_quality.py**: Automated detection of data quality issues
- **eda_utils.py**: Core EDA operations and data loading functions
- **visualizations.py**: Plotly and Seaborn visualization generators

### 4. External Library Integration
- **Profiling Libraries**: pandas-profiling, Sweetviz, AutoViz for automated reports
- **Visualization**: Plotly, Seaborn, Matplotlib for interactive charts
- **Data Processing**: pandas, numpy, scipy for data manipulation
- **Machine Learning**: scikit-learn for preprocessing and scaling
- **AI Integration**: OpenAI GPT-4o for intelligent insights and recommendations
- **Advanced Analytics**: Statistical analysis and business intelligence features

## Data Flow

1. **Data Upload**: Users upload CSV files through the sidebar interface
2. **Data Loading**: Files are processed and stored in session state
3. **Quality Assessment**: Automated detection of 10 common data quality issues
4. **Analysis Options**: Users can choose from multiple analysis approaches:
   - Automated profiling reports
   - Detailed issue analysis with visualizations
   - Interactive exploration tools
   - Data cleaning and preprocessing
5. **Results**: Generated reports, visualizations, and cleaned datasets

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scipy**: Statistical functions

### Specialized Libraries
- **ydata-profiling**: Automated data profiling
- **sweetviz**: Comparative data analysis
- **autoviz**: Automated visualization generation
- **missingno**: Missing data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning preprocessing

## Deployment Strategy

The application is designed for Replit deployment with:
- Streamlit as the web framework (no additional server setup required)
- Python package dependencies managed through requirements
- Session state management for multi-user support
- File upload handling for CSV datasets
- Memory-efficient processing for large datasets

## Changelog

```
Changelog:
- July 07, 2025. Initial setup with basic EDA functionality
- July 07, 2025. Added AI-powered data cleaning with OpenAI GPT-4o integration
- July 07, 2025. Added automated dashboard generator similar to Power BI
- July 07, 2025. Enhanced with intelligent insights and business recommendations
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

### Architecture Decisions

**Problem**: Need for comprehensive EDA tool with automated profiling
**Solution**: Multi-page Streamlit application with modular utility functions
**Rationale**: Streamlit provides rapid development and deployment, while modular design ensures maintainability

**Problem**: Data quality assessment across multiple dimensions
**Solution**: Automated detection system covering 10 common issues
**Rationale**: Systematic approach ensures comprehensive coverage of data quality problems

**Problem**: Visualization variety and interactivity
**Solution**: Integration of multiple visualization libraries (Plotly, Seaborn, specialized tools)
**Rationale**: Different libraries excel at different visualization types, providing comprehensive coverage

**Problem**: Data persistence across pages
**Solution**: Streamlit session state management
**Rationale**: Avoids re-uploading data when navigating between analysis pages

**Problem**: Automated report generation
**Solution**: Integration with industry-standard profiling libraries
**Rationale**: Leverages proven tools rather than rebuilding functionality from scratch
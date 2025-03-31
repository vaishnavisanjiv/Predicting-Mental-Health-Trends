import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.utils import preprocess_data, create_mental_health_summary, plot_feature_distribution, plot_correlation_matrix, generate_recommendations
from src.model import train_model, predict_mental_health, get_feature_importance

# Function to configure plot theme
def configure_plot_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#E0E0E0',
        legend_font_color='#E0E0E0',
        showlegend=True
    )
    fig.update_xaxes(gridcolor='#333333', zerolinecolor='#333333')
    fig.update_yaxes(gridcolor='#333333', zerolinecolor='#333333')
    return fig

# Set page configuration
st.set_page_config(
    page_title="Mental Health Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load the dataset at the beginning
@st.cache_data
def load_data():
    return pd.read_csv("data/mental_health_data.csv")

# Load data once at startup
df = load_data()

# Custom CSS for dark theme and professional styling
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Main content padding */
    .main .block-container {
        padding: 2rem 3rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Card containers - only for forms */
    .stForm {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #333333;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #4F46E5;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #4338CA;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4F46E5;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #333333;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Form inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        color: #E0E0E0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #1E1E1E;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }
    
    /* Plot styling */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    }
</style>
""", unsafe_allow_html=True)

# Get unique courses from the dataset
@st.cache_data
def get_course_options():
    return sorted(df['Course'].unique().tolist())

# Train model if not already trained
@st.cache_resource
def get_model_results():
    return train_model()

# Create vertical navigation using radio buttons in a sidebar
st.sidebar.title("Mental Health Analysis")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Mental Health Assessment", "Personal Insights", "Dataset Analysis"]
)

# Add professional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
A comprehensive mental health analysis platform for students, 
providing data-driven insights and personalized recommendations.

### Features
• Mental health assessment
• Data-driven insights
• Statistical analysis
• Predictive analytics
""")

if page == "Mental Health Assessment":
    st.title("Mental Health Assessment")
    st.markdown("Complete the assessment form for a comprehensive mental health analysis.")
    
    with st.form("mental_health_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=16, max_value=60, value=20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            year = st.selectbox("Year of Study", [1, 2, 3, 4])
            course = st.selectbox("Course", get_course_options())

        with col2:
            st.subheader("Academic & Lifestyle Factors")
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
            study_hours = st.slider("Study Hours per Week", 0, 80, 40)
            engagement = st.slider("Academic Engagement", 1, 10, 7)
            sleep = st.slider("Sleep Quality", 1, 10, 7)
            stress = st.slider("Study Stress Level", 1, 10, 5)

        submit = st.form_submit_button("Generate Analysis")

    if submit:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'CGPA': [cgpa],
            'StudyHoursPerWeek': [study_hours],
            'AcademicEngagement': [engagement],
            'SleepQuality': [sleep],
            'StudyStressLevel': [stress],
            'YearOfStudy': [year]
        })
        
        # Add course dummy variables
        for c in get_course_options():
            input_data[f'Course_{c}'] = 1 if c == course else 0
        
        # Get predictions
        predictions = predict_mental_health(input_data)
        
        st.markdown("### Assessment Results")
        
        # Display results in professional cards
        cols = st.columns(3)
        conditions = {
            'Depression': ('Risk Level', '#4F46E5'),
            'Anxiety': ('Risk Level', '#7C3AED'),
            'PanicAttack': ('Risk Level', '#9333EA')
        }
        
        for i, (condition, pred) in enumerate(predictions.items()):
            prob = pred['probability'] * 100
            risk_level = 'High' if prob > 70 else 'Moderate' if prob > 30 else 'Low'
            with cols[i]:
                st.markdown(f"""
                <div style="padding: 1.5rem; background-color: #1E1E1E; border: 1px solid #333333; border-radius: 4px;">
                    <h3 style="color: {conditions[condition][1]};">{condition}</h3>
                    <h2 style="color: {conditions[condition][1]};">{prob:.1f}%</h2>
                    <p style="color: #E0E0E0;">{conditions[condition][0]}: {risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Professional recommendations
        st.markdown("### Clinical Insights")
        recommendations = []
        if predictions['Depression']['probability'] > 0.3:
            recommendations.append("• Consider professional mental health consultation\n")
        if predictions['Anxiety']['probability'] > 0.3:
            recommendations.append("• Implement stress management strategies\n")
        if predictions['PanicAttack']['probability'] > 0.3:
            recommendations.append("• Develop coping mechanisms for anxiety management\n")
        if sleep < 6:
            recommendations.append("• Prioritize sleep hygiene improvement\n")
        if stress > 7:
            recommendations.append("• Consider stress reduction techniques\n")
        if study_hours > 60:
            recommendations.append("• Evaluate work-life balance\n")
            
        if recommendations:
            st.markdown("\n".join(recommendations))
        else:
            st.markdown("Current indicators suggest stable mental health status. Continue maintaining healthy practices.\n")

elif page == "Personal Insights":
    st.title("Personal Insights")
    st.markdown("Analysis of mental health factors and their correlations.")
    
    st.markdown("### Academic Performance Analysis")
    fig = px.box(df, x="YearOfStudy", y="CGPA",
                color="Depression",
                title="CGPA Distribution by Year")
    fig = configure_plot_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Influencing Factors")
    importance = get_feature_importance()
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance['Depression'].values())[:8],
                y=list(importance['Depression'].keys())[:8],
                orientation='h',
                marker_color='#4F46E5'
            )
        ])
        fig = configure_plot_theme(fig)
        fig.update_layout(
            title="Depression Factors",
            xaxis_title="Impact Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance['Anxiety'].values())[:8],
                y=list(importance['Anxiety'].keys())[:8],
                orientation='h',
                marker_color='#7C3AED'
            )
        ])
        fig = configure_plot_theme(fig)
        fig.update_layout(
            title="Anxiety Factors",
            xaxis_title="Impact Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

else:  # Dataset Analysis
    st.title("Dataset Analysis")
    st.markdown("Statistical analysis of mental health patterns in the student population.")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", len(df))
    with col2:
        st.metric("Depression Prevalence", f"{(df['Depression'].mean() * 100):.1f}%")
    with col3:
        st.metric("Anxiety Prevalence", f"{(df['Anxiety'].mean() * 100):.1f}%")
    with col4:
        st.metric("Panic Attack Incidence", f"{(df['PanicAttack'].mean() * 100):.1f}%")
    
    st.markdown("### Mental Health Distribution Analysis")
    
    viz_tab1, viz_tab2 = st.tabs(["Course Analysis", "Temporal Analysis"])
    
    with viz_tab1:
        course_stats = df.groupby('Course')[['Depression', 'Anxiety', 'PanicAttack']].mean()
        fig = px.bar(course_stats, barmode='group',
                     title="Mental Health Indicators by Course",
                     color_discrete_sequence=['#4F46E5', '#7C3AED', '#9333EA'])
        fig = configure_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        year_stats = df.groupby('YearOfStudy')[['Depression', 'Anxiety', 'PanicAttack']].mean()
        fig = px.line(year_stats, markers=True,
                      title="Mental Health Trends Across Academic Years",
                      color_discrete_sequence=['#4F46E5', '#7C3AED', '#9333EA'])
        fig = configure_plot_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlations
    numeric_cols = ['Age', 'CGPA', 'StudyHoursPerWeek', 'AcademicEngagement', 
                'SleepQuality', 'StudyStressLevel', 'Depression', 'Anxiety', 'PanicAttack']
    corr = df[numeric_cols].corr()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Key Correlations")
        # Find strong correlations (absolute value > 0.3)
        strong_correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_value = corr.iloc[i, j]
                if abs(corr_value) > 0.3 and corr_value != 1.0:  # Exclude self-correlations
                    strong_correlations.append({
                        'factor1': corr.columns[i],
                        'factor2': corr.columns[j],
                        'correlation': corr_value
                    })

        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Display top correlations
        for corr in strong_correlations[:5]:
            direction = "positive" if corr['correlation'] > 0 else "negative"
            strength = "strong" if abs(corr['correlation']) > 0.5 else "moderate"
            st.markdown(f"""
            <div style="padding: 0.5rem; border-left: 3px solid #4F46E5; margin-bottom: 0.5rem;">
                <p style="margin: 0;">
                    <strong>{corr['factor1']} ↔ {corr['factor2']}</strong><br>
                    <span style="color: #888;">{strength.title()} {direction} correlation</span><br>
                    <span style="color: #4F46E5;">{corr['correlation']:.2f}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Key Insights")
        insights = [
            "• Study stress shows strong correlation with anxiety and depression",
            "• Sleep quality inversely correlates with mental health issues",
            "• Academic engagement positively impacts CGPA",
            "• Study hours show moderate correlation with stress levels"
        ]
        for insight in insights:
            st.markdown(f"""
            <div style="padding: 0.5rem; border-left: 3px solid #7C3AED; margin-bottom: 0.5rem;">
                <p style="margin: 0;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)        

    st.markdown("### Dataset Overview")
    # Display options
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search in dataset", "")
    with col2:
        n_rows = st.selectbox("Show rows", [10, 25, 50, 100])

    # Filter dataframe based on search term
    if search:
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        filtered_df = df[mask]
    else:
        filtered_df = df

    # Display the dataframe
    st.dataframe(
        filtered_df.head(n_rows),
        use_container_width=True,
        height=400
    )

    # Download button for the dataset
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="mental_health_data.csv",
        mime="text/csv"
    )
    
    # Show basic statistics
    st.markdown("### Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
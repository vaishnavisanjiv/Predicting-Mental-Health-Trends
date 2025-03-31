import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    """Preprocess the mental health dataset."""
    # Convert categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Convert Year of Study to numeric
    df['YearOfStudy'] = df['YearOfStudy'].str.extract('(\d+)').astype(int)
    
    # Create dummy variables for Course
    course_dummies = pd.get_dummies(df['Course'], prefix='Course')
    df = pd.concat([df, course_dummies], axis=1)
    
    return df

def create_mental_health_summary(df):
    """Create a summary of mental health statistics."""
    mental_health_cols = ['Depression', 'Anxiety', 'PanicAttack']
    summary = df[mental_health_cols].agg(['mean', 'sum']).round(3)
    return summary

def plot_feature_distribution(df, feature, title):
    """Create a distribution plot for a feature."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, bins=20)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Count')
    return plt.gcf()

def plot_correlation_matrix(df):
    """Create a correlation matrix heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    return plt.gcf()

def generate_recommendations(results):
    """Generate personalized recommendations based on assessment results."""
    recommendations = []
    
    # Sleep Quality Recommendations
    if results['Sleep Quality'] <= 2:
        recommendations.append({
            'category': 'Sleep Quality',
            'priority': 'High',
            'suggestions': [
                'Try to maintain a consistent sleep schedule',
                'Create a relaxing bedtime routine',
                'Limit screen time before bed',
                'Consider using sleep tracking apps'
            ]
        })
    
    # Study Stress Recommendations
    if results['Study Stress'] >= 4:
        recommendations.append({
            'category': 'Study Stress',
            'priority': 'High',
            'suggestions': [
                'Implement regular study breaks',
                'Practice stress management techniques',
                'Seek academic support',
                'Consider time management workshops'
            ]
        })
    
    # Academic Performance Recommendations
    if results['CGPA'] < 2.5:
        recommendations.append({
            'category': 'Academic Performance',
            'priority': 'High',
            'suggestions': [
                'Meet with academic advisors',
                'Join study groups',
                'Utilize campus tutoring services',
                'Develop a study schedule'
            ]
        })
    
    # Mental Health Support Recommendations
    if results['Depression'] or results['Anxiety'] or results['Panic Attacks']:
        if not results['Has Support']:
            recommendations.append({
                'category': 'Mental Health Support',
                'priority': 'High',
                'suggestions': [
                    'Reach out to campus counseling services',
                    'Connect with support groups',
                    'Talk to trusted friends or family',
                    'Consider speaking with a mental health professional'
                ]
            })
    
    return recommendations 
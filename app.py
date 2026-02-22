import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Set page config
st.set_page_config(
    page_title="HeartGuard - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external stylesheet if present (keeps app.py tidy and makes visual updates easier)

try:
    with open("assets/styles.css", "r", encoding="utf-8") as _css:
        st.markdown(f"<style>{_css.read()}</style>", unsafe_allow_html=True)
except Exception:
    # If the external stylesheet isn't available for some reason, fall back to a minimal inline style
    st.markdown("""
    <style>
        .main-header { text-align:left; padding:0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and preprocess the heart disease dataset."""
    try:
        # Try to load data from URL first
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            df = pd.read_csv(StringIO(response.text), names=columns, na_values='?')
        else:
            # Fallback: create sample data if URL fails
            st.warning("Unable to load external data. Using sample dataset.")
            return create_sample_data()
        
        # Convert target to binary
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        df[['ca', 'thal']] = imputer.fit_transform(df[['ca', 'thal']])
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], bins=[20, 40, 60, 80], labels=[0, 1, 2])
        df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 200], labels=[0, 1, 2])
        
        df['age_group'] = df['age_group'].astype(int)
        df['bp_category'] = df['bp_category'].astype(int)
        
        # String mappings for display
        df['sex_str'] = df['sex'].map({1: 'Male', 0: 'Female'})
        df['target_str'] = df['target'].map({1: 'Heart Disease', 0: 'Healthy'})
        df['cp_str'] = df['cp'].map({
            1: 'Typical Angina', 
            2: 'Atypical Angina', 
            3: 'Non-anginal Pain', 
            4: 'Asymptomatic'
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using sample data instead.")
        return create_sample_data()

def create_sample_data():
    """Create sample data if external data loading fails."""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([1, 2, 3, 4], n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 564, n_samples),
        'fbs': np.random.choice([0, 1], n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.randint(71, 202, n_samples),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.choice([1, 2, 3], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples),
        'thal': np.random.choice([3, 6, 7], n_samples),
    }
    
    # Create target variable based on some logic
    df = pd.DataFrame(data)
    df['target'] = ((df['age'] > 55) | 
                   (df['cp'] == 4) | 
                   (df['trestbps'] > 140) | 
                   (df['chol'] > 240) | 
                   (df['exang'] == 1)).astype(int)
    
    # Add engineered features
    df['age_group'] = pd.cut(df['age'], bins=[20, 40, 60, 80], labels=[0, 1, 2]).astype(int)
    df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 200], labels=[0, 1, 2]).astype(int)
    
    # String mappings
    df['sex_str'] = df['sex'].map({1: 'Male', 0: 'Female'})
    df['target_str'] = df['target'].map({1: 'Heart Disease', 0: 'Healthy'})
    df['cp_str'] = df['cp'].map({
        1: 'Typical Angina', 
        2: 'Atypical Angina', 
        3: 'Non-anginal Pain', 
        4: 'Asymptomatic'
    })
    
    return df

# Train and save model
@st.cache_resource
def train_model():
    """Train the machine learning model."""
    try:
        df = load_data()
        feature_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
            'age_group', 'bp_category'
        ]
        
        X = df[feature_columns]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            min_samples_split=5,
            max_features='sqrt'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        importances = model.feature_importances_
        feature_names = [
            'Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 'Blood Sugar',
            'ECG', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'Slope',
            'Vessels', 'Thalassemia', 'Age Group', 'BP Category'
        ]
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return model, accuracy, cm, fpr, tpr, roc_auc, feature_importance_df
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None, None, None, None, None

# Create input form
def input_form():
    """Create the user input form."""
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("🏥 Patient Health Information")
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🧍 Personal Information")
            age = st.slider("Age (Years)", 20, 100, 50, key='age')
            sex = st.radio("Sex", ["Male", "Female"], index=0, key='sex')
            
            st.markdown("#### 💓 Heart Metrics")
            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina", 
                "Atypical Angina", 
                "Non-anginal Pain", 
                "Asymptomatic"
            ], index=2, key='cp')
            
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120, key='trestbps')
            thalach = st.slider("Max Heart Rate Achieved", 70, 220, 150, key='thalach')
            
        with col2:
            st.markdown("#### 🩸 Blood Analysis")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200, key='chol')
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0, key='fbs')
            
            st.markdown("#### 📊 ECG & Exercise")
            restecg = st.selectbox("Resting ECG Results", [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ], index=0, key='restecg')
            
            exang = st.radio("Exercise Induced Angina", ["No", "Yes"], index=0, key='exang')
            
        with col3:
            st.markdown("#### 🏥 Medical Details")
            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1, key='oldpeak')
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping",
                "Flat",
                "Downsloping"
            ], index=0, key='slope')
            
            ca = st.slider("Number of Major Vessels Colored (0-3)", 0, 3, 0, key='ca')
            thal = st.selectbox("Thalassemia Type", [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ], index=0, key='thal')
        
        submit_button = st.form_submit_button("🚀 Analyze Heart Health", use_container_width=True)
    
    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'submit': submit_button
    }

# Preprocess inputs
def preprocess_inputs(inputs):
    """Convert user inputs to model features."""
    sex = 1 if inputs['sex'] == "Male" else 0
    
    cp_mapping = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4
    }
    cp = cp_mapping[inputs['cp']]
    
    fbs = 1 if inputs['fbs'] == "Yes" else 0
    
    restecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_mapping[inputs['restecg']]
    
    exang = 1 if inputs['exang'] == "Yes" else 0
    
    slope_mapping = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }
    slope = slope_mapping[inputs['slope']]
    
    thal_mapping = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    }
    thal = thal_mapping[inputs['thal']]
    
    age = inputs['age']
    trestbps = inputs['trestbps']
    age_group = 1 if 40 <= age <= 60 else (2 if age > 60 else 0)
    bp_category = 1 if 120 < trestbps <= 140 else (2 if trestbps > 140 else 0)
    
    feature_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': inputs['chol'],
        'fbs': fbs,
        'restecg': restecg,
        'thalach': inputs['thalach'],
        'exang': exang,
        'oldpeak': inputs['oldpeak'],
        'slope': slope,
        'ca': inputs['ca'],
        'thal': thal,
        'age_group': age_group,
        'bp_category': bp_category
    }
    
    features_df = pd.DataFrame([feature_dict])
    
    return features_df

# Display results
def display_results(prediction, probability, model, features):
    """Display prediction results with visualizations."""
    if prediction == 1:
        with st.container():
            st.markdown(
                f"<div class='result-card'>"
                f"<h2 class='risk'>⚠️ Heart Disease Risk Detected</h2>"
                f"<p style='font-size: 18px;'>Risk Probability: <span class='risk'>{probability:.1%}</span></p>"
                f"<div class='risk-meter'>"
                f"<div class='risk-indicator' style='left:{probability*100}%'></div>"
                f"</div>"
                f"<div class='risk-labels'>"
                f"<span>Low Risk</span>"
                f"<span>Moderate</span>"
                f"<span>High Risk</span>"
                f"</div>"
                f"<p style='margin-top: 20px;'>Our analysis indicates a significant risk of heart disease. Please consult a cardiologist for a comprehensive evaluation.</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with st.expander("📋 Detailed Health Recommendations", expanded=True):
            st.markdown("""
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff">🩺 Medical Consultation</h4>
                <ul>
                    <li>Schedule an appointment with a cardiologist immediately</li>
                    <li>Request a full cardiac workup including stress test and echocardiogram</li>
                    <li>Consider additional tests like coronary angiography if recommended</li>
                </ul>
            </div>
            
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff;">📊 Health Monitoring</h4>
                <ul>
                    <li>Monitor blood pressure twice daily and maintain a log</li>
                    <li>Track cholesterol levels monthly with lipid profile tests</li>
                    <li>Keep a symptom diary noting chest discomfort or shortness of breath</li>
                    <li>Regular follow-up appointments with healthcare provider</li>
                </ul>
            </div>
            
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff;">🍎 Lifestyle Changes</h4>
                <ul>
                    <li>Adopt a heart-healthy diet (Mediterranean diet recommended)</li>
                    <li>Begin light exercise (walking) after medical clearance</li>
                    <li>Eliminate smoking and reduce alcohol consumption</li>
                    <li>Practice stress management techniques daily (meditation, deep breathing)</li>
                    <li>Maintain a healthy weight and get adequate sleep (7-9 hours)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(
                f"<div class='result-card'>"
                f"<h2 class='healthy'>✅ Healthy Heart Profile</h2>"
                f"<p style='font-size: 18px;'>Healthy Probability: <span class='healthy'>{1-probability:.1%}</span></p>"
                f"<div class='risk-meter'>"
                f"<div class='risk-indicator' style='left:{probability*100}%'></div>"
                f"</div>"
                f"<div class='risk-labels'>"
                f"<span>Low Risk</span>"
                f"<span>Moderate</span>"
                f"<span>High Risk</span>"
                f"</div>"
                f"<p style='margin-top: 20px;'>Your heart health metrics indicate a low risk of heart disease. Continue your healthy habits!</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with st.expander("📋 Heart Health Maintenance Tips", expanded=True):
            st.markdown("""
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff;">💪 Preventive Care</h4>
                <ul>
                    <li>Continue regular cardiovascular exercise (150 minutes/week)</li>
                    <li>Maintain annual heart health checkups</li>
                    <li>Monitor key metrics: blood pressure, cholesterol, blood sugar</li>
                    <li>Stay up to date with preventive screenings</li>
                </ul>
            </div>
            
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff;">🥗 Nutrition Guidance</h4>
                <ul>
                    <li>Focus on fruits, vegetables, whole grains, and lean proteins</li>
                    <li>Limit saturated fats, trans fats, and sodium intake</li>
                    <li>Include omega-3 rich foods like salmon, walnuts, and flaxseeds</li>
                    <li>Stay hydrated and limit processed foods</li>
                </ul>
            </div>
            
            <div class='recommendation-card'>
                <h4 style="color: #4b75ff;">😌 Wellness Practices</h4>
                <ul>
                    <li>Practice stress-reduction techniques (meditation, yoga)</li>
                    <li>Maintain healthy sleep patterns (7-9 hours/night)</li>
                    <li>Avoid tobacco products and limit alcohol intake</li>
                    <li>Stay socially connected and maintain mental health</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.subheader("🔑 Key Contributing Factors")
    st.divider()
    
    if model is not None:
        importances = model.feature_importances_
        feature_names = [
            'Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 'Blood Sugar',
            'ECG', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'Slope',
            'Vessels', 'Thalassemia', 'Age Group', 'BP Category'
        ]
        
        top_features_idx = importances.argsort()[::-1][:8]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importances = [importances[i] for i in top_features_idx]
        
        fig = go.Figure(go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Reds',
                showscale=True
            ),
            text=[f"{imp:.3f}" for imp in top_importances],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Most Important Factors in Your Prediction',
            height=400,
            margin=dict(l=150, r=50, t=50, b=50),
            xaxis_title='Feature Importance',
            yaxis_title='Health Factor',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key='top_feature_importance')

# Data exploration section
def show_data_exploration(df, cm, fpr, tpr, roc_auc, feature_importance_df):
    """Display data exploration visualizations."""
    if df is None:
        st.error("Data not available for exploration.")
        return
        
    st.subheader("📊 Heart Disease Data Analysis")
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Heart Disease Cases", len(df[df['target'] == 1]))
    with col3:
        st.metric("Healthy Cases", len(df[df['target'] == 0]))
    with col4:
        st.metric("Average Age", f"{df['age'].mean():.1f} years")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=['Healthy', 'Heart Disease'],
                   yticklabels=['Healthy', 'Heart Disease'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Condition')
        ax.set_ylabel('Actual Condition')
        ax.set_title('Model Prediction Accuracy')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### ROC Curve Analysis")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#ff4b4b', width=3)
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], 
            mode='lines', 
            line=dict(color='gray', dash='dash'),
            name='Random Chance (AUC = 0.5)'
        ))
        roc_fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.6, y=0.2)
        )
        st.plotly_chart(roc_fig, use_container_width=True, key='roc_explore')
    
    st.markdown("#### Feature Importance Analysis")
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                orientation='h', color='Importance',
                color_continuous_scale='Reds',
                labels={'Importance': 'Feature Importance Score'})
    fig.update_layout(
        height=500, 
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='feat_importance_explore')
    
    with st.expander("🔧 Technical Model Details"):
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        - **Number of Trees:** 200
        - **Max Depth:** 10
        - **Min Samples Split:** 5
        - **Max Features:** sqrt
        - **Class Weight:** Balanced
        
        **Training Details:**
        - **Dataset:** Cleveland Heart Disease Dataset (UCI)
        - **Training/Test Split:** 80/20
        - **Cross-validation:** Stratified sampling
        - **Missing Value Handling:** Most frequent imputation
        
        **Feature Engineering:**
        - Age grouping (young, middle-aged, elderly)
        - Blood pressure categorization
        - Original features + engineered features
        """)
        
    # Additional visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Age Distribution by Heart Condition")
        fig = px.histogram(df, x='age', color='target_str', 
                          nbins=20, barmode='overlay', opacity=0.7,
                          color_discrete_sequence=['#4caf50', '#ff4b4b'])
        fig.update_layout(
            legend_title_text='Condition',
            xaxis_title='Age (Years)',
            yaxis_title='Number of Patients',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, key='age_dist')
        
    with col2:
        st.markdown("#### Heart Disease Prevalence by Gender")
        gender_counts = df.groupby(['sex_str', 'target_str']).size().unstack(fill_value=0)
        fig = px.bar(gender_counts, barmode='group',
                    color_discrete_sequence=['#4caf50', '#ff4b4b'])
        fig.update_layout(
            xaxis_title='Gender',
            yaxis_title='Number of Patients',
            legend_title_text='Condition',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, key='gender_prevalence')
    
    st.markdown("#### Cholesterol vs Blood Pressure by Heart Condition")
    fig = px.scatter(df, x='chol', y='trestbps', color='target_str',
                    color_discrete_sequence=['#4caf50', '#ff4b4b'],
                    hover_data=['age', 'thalach'],
                    labels={'chol': 'Cholesterol (mg/dl)', 
                           'trestbps': 'Resting Blood Pressure (mm Hg)'})
    fig.update_layout(
        legend_title_text='Condition',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key='chol_vs_bp')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Chest Pain Types Distribution")
        cp_counts = df['cp_str'].value_counts()
        fig = px.pie(cp_counts, values=cp_counts.values, names=cp_counts.index,
                    color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True, key='cp_pie')
    
    with col2:
        st.markdown("#### Max Heart Rate by Age and Condition")
        fig = px.scatter(df, x='age', y='thalach', color='target_str',
                        color_discrete_sequence=['#4caf50', '#ff4b4b'],
                        labels={'age': 'Age (Years)', 
                               'thalach': 'Max Heart Rate'})
        fig.update_layout(
            legend_title_text='Condition',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True, key='maxhr_age')

# Model information section
def show_model_info(model, accuracy, cm, fpr, tpr, roc_auc, feature_importance_df):
    """Display model performance metrics and information."""
    if model is None:
        st.error("Model not available.")
        return
        
    st.subheader("🤖 Model Performance & Information")
    st.divider()
    
    st.markdown("#### Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("ROC AUC Score", f"{roc_auc:.3f}")
    with col3:
        st.metric("Number of Trees", model.n_estimators)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=['Healthy', 'Heart Disease'],
                   yticklabels=['Healthy', 'Heart Disease'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Condition')
        ax.set_ylabel('Actual Condition')
        ax.set_title('Model Prediction Accuracy')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### ROC Curve")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#ff4b4b', width=3)
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], 
            mode='lines', 
            line=dict(color='gray', dash='dash'),
            name='Random Chance (AUC = 0.5)'
        ))
        roc_fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.6, y=0.2)
        )
        st.plotly_chart(roc_fig, use_container_width=True, key='roc_model')
    
    st.markdown("#### Model Parameters")
    st.json({
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "random_state": model.random_state,
        "class_weight": str(model.class_weight),
        "min_samples_split": model.min_samples_split,
        "max_features": model.max_features
    })
    
    st.markdown("#### Feature Importance")
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', 
                orientation='h', color='Importance',
                color_continuous_scale='Reds',
                labels={'Importance': 'Feature Importance Score'})
    fig.update_layout(
        height=500, 
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, key='feat_importance_model')

# Main app function
def main():
    """Main application function."""
    # Header: logo + title
    try:
        col1, col2 = st.columns([1, 8])
        with col1:
            st.image("assets/logo.svg", width=84)
        with col2:
            st.markdown(
                """
                <div class='main-header'>
                    <h1 style='margin:0'>❤️ HeartGuard - Heart Disease Prediction</h1>
                    <p style='margin:0; color: #666;'>Advanced ML-powered cardiovascular risk assessment</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception:
        # Fallback simple header
        st.markdown("<h1>❤️ HeartGuard - Heart Disease Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(90deg, #237c95cc 0%, #5ac8facc 100%); border-radius: 10px; box-shadow: 0 2px 8px rgba(149, 35, 94, 0.8);'>
        <p style='font-size: 18px; margin-bottom: 10px;'>
            This application uses machine learning to assess your risk of heart disease based on health metrics.
        </p>
        <p style='font-size: 14px; color: white;'>
            Trained on the Cleveland Heart Disease dataset from UCI Machine Learning Repository.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏥 Heart Disease Prediction", "📊 Data Exploration", "🤖 Model Information"])
    
    # Load data and train model
    df = load_data()
    model_results = train_model()
    model, accuracy, cm, fpr, tpr, roc_auc, feature_importance_df = model_results
    
    with tab1:
        st.markdown("### 📋 Enter Patient Information")
        inputs = input_form()
        
        if inputs['submit']:
            if model is not None:
                features = preprocess_inputs(inputs)
                
                with st.spinner('🔍 Analyzing your heart health metrics...'):
                    time.sleep(1.5)
                    try:
                        prediction = model.predict(features)[0]
                        probability = model.predict_proba(features)[0][1]
                        
                        display_results(prediction, probability, model, features)
                        
                        with st.expander("📝 Input Summary", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Personal Information:**")
                                st.write(f"- Age: {inputs['age']} years")
                                st.write(f"- Sex: {inputs['sex']}")
                                st.write(f"- Chest Pain Type: {inputs['cp']}")
                                
                                st.write("**Vital Signs:**")
                                st.write(f"- Resting BP: {inputs['trestbps']} mm Hg")
                                st.write(f"- Max Heart Rate: {inputs['thalach']} bpm")
                                st.write(f"- Cholesterol: {inputs['chol']} mg/dl")
                                
                            with col2:
                                st.write("**Medical History:**")
                                st.write(f"- Fasting Blood Sugar >120: {inputs['fbs']}")
                                st.write(f"- Resting ECG: {inputs['restecg']}")
                                st.write(f"- Exercise Angina: {inputs['exang']}")
                                
                                st.write("**Additional Metrics:**")
                                st.write(f"- ST Depression: {inputs['oldpeak']}")
                                st.write(f"- Slope: {inputs['slope']}")
                                st.write(f"- Major Vessels: {inputs['ca']}")
                                st.write(f"- Thalassemia: {inputs['thal']}")
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Model is not available. Please try refreshing the page.")
    
    with tab2:
        if all(item is not None for item in [df, cm, fpr, tpr, roc_auc, feature_importance_df]):
            show_data_exploration(df, cm, fpr, tpr, roc_auc, feature_importance_df)
        else:
            st.error("Data not available for exploration.")
    
    with tab3:
        if all(item is not None for item in [model, accuracy, cm, fpr, tpr, roc_auc, feature_importance_df]):
            show_model_info(model, accuracy, cm, fpr, tpr, roc_auc, feature_importance_df)
        else:
            st.error("Model information not available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;'>
        <h4 style="color: #4b75ff;">⚠️ Medical Disclaimer</h4>
        <p style='color: #666; font-size: 14px;'>
            This tool provides risk assessment only and is not a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
            with any questions you may have regarding a medical condition.
        </p>
        <hr style='margin: 20px 0; border: none; border-top: 1px solid #ddd;'>
        <p style='color: #888; font-size: 12px;'>
            HeartGuard v2.0 | Powered by Machine Learning | Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
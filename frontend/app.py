import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Animal Type Classification System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_classify(image_file, species, animal_name=None):
    """Upload image and get classification results."""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        data = {
            "species": species,
            "animal_name": animal_name or f"Animal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/upload", 
            files=files, 
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_classification_history():
    """Get classification history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/history", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Could not fetch history: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching history: {str(e)}"}

def display_trait_scorecard(trait_scores):
    """Display detailed trait scorecard."""
    if not trait_scores:
        st.warning("No trait scores available")
        return
    
    # Create metrics cards
    cols = st.columns(3)
    
    trait_items = list(trait_scores.items())
    for i, (trait_name, trait_data) in enumerate(trait_items):
        col_idx = i % 3
        
        with cols[col_idx]:
            score = trait_data.get('score', 0)
            value = trait_data.get('value', 0)
            ideal_range = trait_data.get('ideal_range', 'N/A')
            description = trait_data.get('description', '')
            
            # Color based on score
            if score >= 80:
                color = "🟢"
            elif score >= 60:
                color = "🟡"
            else:
                color = "🔴"
            
            st.metric(
                label=f"{color} {trait_name.replace('_', ' ').title()}",
                value=f"{score:.1f}/100",
                delta=f"Value: {value:.3f}"
            )
            
            with st.expander(f"Details - {trait_name.replace('_', ' ').title()}"):
                st.write(f"**Description:** {description}")
                st.write(f"**Current Value:** {value:.3f}")
                st.write(f"**Ideal Range:** {ideal_range}")
                st.write(f"**Score:** {score:.1f}/100")

def create_radar_chart(trait_scores):
    """Create radar chart for trait visualization."""
    if not trait_scores:
        return None
    
    traits = []
    scores = []
    
    for trait_name, trait_data in trait_scores.items():
        traits.append(trait_name.replace('_', ' ').title())
        scores.append(trait_data.get('score', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=traits,
        fill='toself',
        name='Animal Scores',
        line_color='rgb(0, 123, 255)',
        fillcolor='rgba(0, 123, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Trait Performance Radar Chart",
        height=400
    )
    
    return fig

def create_probability_chart(prob_distribution):
    """Create probability distribution chart."""
    if not prob_distribution:
        return None
    
    classes = list(prob_distribution.keys())
    probs = [prob_distribution[cls] * 100 for cls in classes]
    
    colors = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors[:len(classes)],
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Productivity Classification Probabilities",
        xaxis_title="Productivity Class",
        yaxis_title="Probability (%)",
        height=300
    )
    
    return fig

def main():
    # Header
    st.title("🐄 Animal Type Classification System")
    st.markdown("### AI-Powered Productivity Assessment for Cows and Buffaloes")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start the backend server first.")
        st.code("cd backend && python main.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Classification", "📈 History", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.header("System Status")
        st.success("✅ Backend Connected")
        st.info(f"API: {API_BASE_URL}")
    
    # Main content based on selected page
    if page == "🏠 Home":
        display_home_page()
    elif page == "📊 Classification":
        display_classification_page()
    elif page == "📈 History":
        display_history_page()
    elif page == "ℹ️ About":
        display_about_page()

def display_home_page():
    """Display the home page."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Animal Type Classification System
        
        This system uses **computer vision** and **machine learning** to assess the productivity potential 
        of cows and buffaloes based on their body structure traits.
        
        ### Key Features:
        - 📸 **Image Analysis**: Upload photos for automated feature extraction
        - 🎯 **AI Classification**: Get productivity assessments (High/Medium/Low)
        - 📊 **Detailed Scoring**: View trait-wise performance scorecards
        - 📈 **Visual Analytics**: Interactive charts and radar plots
        - 🗄️ **History Tracking**: Store and review past assessments
        
        ### How It Works:
        1. **Upload** a clear side-view image of your animal
        2. **Select** the species (cow or buffalo)
        3. **Get Results** including productivity classification and detailed trait analysis
        4. **Review** recommendations for animal management
        
        ### Measured Traits:
        - Body length and proportions
        - Chest depth and capacity
        - Leg structure and soundness
        - Body condition scoring
        - Udder placement assessment
        - Overall structural conformation
        """)
        
        if st.button("🚀 Start Classification", type="primary", use_container_width=True):
            st.session_state.page = "📊 Classification"
            st.rerun()
    
    with col2:
        st.markdown("### Quick Stats")
        
        # Get some basic stats
        try:
            response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                st.metric("Total Classifications", stats.get('total_classifications', 0))
                st.metric("High Productivity", stats.get('high_productivity_count', 0))
                st.metric("Average Score", f"{stats.get('average_score', 0):.1f}/100")
            else:
                st.metric("Total Classifications", "N/A")
        except:
            st.metric("Total Classifications", "N/A")
        
        st.markdown("### Recent Activity")
        st.info("Upload your first animal image to get started!")

def display_classification_page():
    """Display the classification page."""
    
    st.header("📊 Animal Classification")
    
    # Upload section
    st.subheader("1. Upload Animal Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear side-view image of the animal"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("2. Animal Details")
        
        species = st.selectbox(
            "Select Species",
            ["cow", "buffalo"],
            help="Choose the animal species"
        )
        
        animal_name = st.text_input(
            "Animal Name (Optional)",
            placeholder="e.g., Bella, Thunder",
            help="Give your animal a name for easier tracking"
        )
        
        st.markdown("### Image Guidelines:")
        st.markdown("""
        - ✅ Clear side view
        - ✅ Good lighting
        - ✅ Minimal background
        - ✅ Full body visible
        - ❌ Avoid blurry images
        - ❌ Avoid extreme angles
        """)
    
    # Classification button
    if uploaded_file is not None:
        if st.button("🔍 Classify Animal", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and extracting features..."):
                result = upload_and_classify(uploaded_file, species, animal_name)
            
            if "error" in result:
                st.error(f"Classification failed: {result['error']}")
            else:
                display_classification_results(result)

def display_classification_results(result):
    """Display classification results."""
    
    st.success("✅ Classification completed successfully!")
    
    # Main results
    st.subheader("🎯 Classification Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        productivity_class = result['classification']['productivity_class']
        confidence = result['classification']['confidence']
        
        # Color coding
        if productivity_class == 'High':
            color = "🟢"
        elif productivity_class == 'Medium':
            color = "🟡"
        else:
            color = "🔴"
        
        st.metric(
            label="Productivity Class",
            value=f"{color} {productivity_class}",
            delta=f"Confidence: {confidence:.1%}"
        )
    
    with col2:
        overall_score = result['classification']['overall_score']
        st.metric(
            label="Overall Score",
            value=f"{overall_score}/100",
            delta="Based on all traits"
        )
    
    with col3:
        species = result['species'].title()
        st.metric(
            label="Species",
            value=species,
            delta=result['animal_name']
        )
    
    with col4:
        timestamp = result['timestamp']
        date_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        st.metric(
            label="Analysis Date",
            value=date_str,
            delta="Completed"
        )
    
    # Probability distribution
    if 'probability_distribution' in result['classification']:
        st.subheader("📈 Probability Distribution")
        prob_chart = create_probability_chart(result['classification']['probability_distribution'])
        if prob_chart:
            st.plotly_chart(prob_chart, use_container_width=True)
    
    # Trait scorecard
    if 'trait_scores' in result['classification']:
        st.subheader("📋 Detailed Trait Scorecard")
        
        # Radar chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            radar_chart = create_radar_chart(result['classification']['trait_scores'])
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
        
        with col2:
            # Trait details
            display_trait_scorecard(result['classification']['trait_scores'])
    
    # Recommendations
    if 'recommendations' in result['classification']:
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(result['classification']['recommendations'], 1):
            st.write(f"{i}. {recommendation}")
    
    # Raw data (expandable)
    with st.expander("🔍 View Technical Details"):
        st.json(result)

def display_history_page():
    """Display the history page."""
    
    st.header("📈 Classification History")
    
    # Get history data
    with st.spinner("Loading classification history..."):
        history_data = get_classification_history()
    
    if "error" in history_data:
        st.error(f"Could not load history: {history_data['error']}")
        return
    
    history = history_data.get('history', [])
    
    if not history:
        st.info("No classification history available. Upload some images to get started!")
        return
    
    # Summary stats
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_count = len(history)
    high_count = sum(1 for h in history if h.get('classification', {}).get('productivity_class') == 'High')
    medium_count = sum(1 for h in history if h.get('classification', {}).get('productivity_class') == 'Medium')
    low_count = total_count - high_count - medium_count
    
    with col1:
        st.metric("Total Classifications", total_count)
    with col2:
        st.metric("High Productivity", high_count, f"{high_count/total_count:.1%}" if total_count > 0 else "0%")
    with col3:
        st.metric("Medium Productivity", medium_count, f"{medium_count/total_count:.1%}" if total_count > 0 else "0%")
    with col4:
        st.metric("Low Productivity", low_count, f"{low_count/total_count:.1%}" if total_count > 0 else "0%")
    
    # History table
    st.subheader("📋 Recent Classifications")
    
    # Prepare data for display
    display_data = []
    for item in history:
        display_data.append({
            'Animal Name': item.get('animal_name', 'Unknown'),
            'Species': item.get('species', '').title(),
            'Productivity Class': item.get('classification', {}).get('productivity_class', 'Unknown'),
            'Overall Score': f"{item.get('classification', {}).get('overall_score', 0):.1f}/100",
            'Confidence': f"{item.get('classification', {}).get('confidence', 0):.1%}",
            'Date': datetime.fromisoformat(item.get('timestamp', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M') if item.get('timestamp') else 'Unknown'
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True)
    
    # Trends chart
    if len(history) > 1:
        st.subheader("📈 Trends Over Time")
        
        # Prepare time series data
        dates = []
        scores = []
        classes = []
        
        for item in history:
            if item.get('timestamp'):
                dates.append(datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')))
                scores.append(item.get('classification', {}).get('overall_score', 0))
                classes.append(item.get('classification', {}).get('productivity_class', 'Unknown'))
        
        if dates and scores:
            fig = px.line(
                x=dates, 
                y=scores,
                title="Overall Scores Over Time",
                labels={'x': 'Date', 'y': 'Overall Score'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def display_about_page():
    """Display the about page."""
    
    st.header("ℹ️ About the ATC System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Animal Type Classification (ATC) System
        
        ### Overview
        The ATC system is a prototype for evaluating cows and buffaloes based on their body structure traits 
        using advanced computer vision and machine learning techniques.
        
        ### Technology Stack
        - **Frontend**: Streamlit (Python web framework)
        - **Backend**: FastAPI (REST API)
        - **Computer Vision**: OpenCV for image processing
        - **Machine Learning**: Random Forest classifier (scikit-learn)
        - **Database**: SQLite for data storage
        - **Visualization**: Plotly for interactive charts
        
        ### Features Extracted
        1. **Body Length Ratio**: Proportion of body length to height
        2. **Chest Depth Ratio**: Chest width relative to body size
        3. **Leg Length Ratio**: Leg proportions for mobility assessment
        4. **Body Condition Score**: Overall health and condition
        5. **Structural Soundness**: Structural integrity assessment
        6. **Udder Placement**: Reproductive and milking efficiency indicators
        
        ### Classification Classes
        - **High Productivity**: Animals with optimal body structure for production
        - **Medium Productivity**: Animals with good structure and moderate limitations
        - **Low Productivity**: Animals with structural issues affecting productivity
        
        ### Model Training
        The system uses a Random Forest classifier trained on simulated data representing 
        various animal body measurements and their correlation with productivity outcomes.
        """)
    
    with col2:
        st.markdown("### System Information")
        st.info(f"**API Endpoint**: {API_BASE_URL}")
        st.info("**Version**: 1.0.0")
        st.info("**Status**: Prototype")
        
        st.markdown("### Model Performance")
        try:
            # Try to get model metadata
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Model Loaded")
                st.success("✅ Image Processor Ready")
                st.success("✅ Database Connected")
            else:
                st.warning("⚠️ Service Status Unknown")
        except:
            st.error("❌ Cannot Connect to Backend")
        
        st.markdown("### Future Enhancements")
        st.markdown("""
        - Deep learning CNN models
        - Real-time video processing
        - Breed-specific classification
        - Mobile app integration
        - Advanced trait extraction
        """)

if __name__ == "__main__":
    main()
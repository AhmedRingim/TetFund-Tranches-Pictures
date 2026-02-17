import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import io
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import tempfile
import time
import gdown
warnings.filterwarnings('ignore')

# Model paths
MODEL_PATH = "models/tranches_model_fp16.tflite"
FILE_ID = "1oy7PzH1RQKVY3_hyxhBfYNn2phdKzWTg"

@st.cache_resource
def load_tranche_model():
    """Load the TFLite model for tranche payment classification"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download model if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Store model info in session state
        st.session_state.input_details = input_details
        st.session_state.output_details = output_details
        st.session_state.input_shape = input_details[0]['shape']
        
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_with_tflite(interpreter, img_array):
    """Make prediction using TFLite model"""
    try:
        # Get input and output details
        input_details = st.session_state.input_details
        output_details = st.session_state.output_details
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # For binary classification, output is probability of class 1
        prediction = output_data[0][0]
        predicted_class = int(prediction > 0.5)
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        return predicted_class, confidence, prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Page configuration
st.set_page_config(
    page_title="Tranche Payment Classifier",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        font-weight: semi-bold;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border-left: 0.5rem solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #10B981;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
    }
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    .logo-image {
        max-width: 100px;
        max-height: 100px;
        object-fit: contain;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Final_Tranche_Payment', 'Second_Tranche_Payment']
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'input_details' not in st.session_state:
    st.session_state.input_details = None
if 'output_details' not in st.session_state:
    st.session_state.output_details = None
if 'input_shape' not in st.session_state:
    st.session_state.input_shape = None

# Constants
IMAGE_WIDTH = 400
GALLERY_WIDTH = 250

# Helper function for logo
def get_logo_base64():
    """Convert logo to base64 for embedding"""
    try:
        with open("logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Image preprocessing function
def preprocess_image(uploaded_image, target_size=(224, 224)):
    """Preprocess uploaded image for model prediction"""
    try:
        img = Image.open(uploaded_image)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

# PDF Report Generator Class
class PDFReportGenerator:
    def __init__(self, filename, pagesize=A4):
        self.filename = filename
        self.pagesize = pagesize
        self.styles = getSampleStyleSheet()
        self.doc = SimpleDocTemplate(filename, pagesize=pagesize)
        self.story = []
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom styles for the PDF"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E3A8A'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563EB'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#374151'),
            spaceAfter=6
        )
    
    def add_logo(self):
        """Add logo to PDF"""
        try:
            if os.path.exists("logo.png"):
                logo = Image.open("logo.png")
                img_buffer = io.BytesIO()
                logo.thumbnail((80, 80))
                logo.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                logo_img = RLImage(img_buffer, width=1.2*inch, height=1.2*inch)
                
                # Center the logo
                logo_table = Table([[logo_img]], colWidths=[7*inch])
                logo_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                self.story.append(logo_table)
                self.story.append(Spacer(1, 0.2*inch))
        except:
            # If logo not found, just continue without it
            pass
    
    def add_title(self, text):
        self.story.append(Paragraph(text, self.title_style))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_heading(self, text):
        self.story.append(Paragraph(text, self.heading_style))
    
    def add_paragraph(self, text):
        self.story.append(Paragraph(text, self.normal_style))
    
    def add_spacer(self, height=0.2):
        self.story.append(Spacer(1, height*inch))
    
    def add_table(self, data):
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB'))
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_prediction_result(self, image, results):
        """Add prediction result with image and details"""
        img_buffer = io.BytesIO()
        image.thumbnail((300, 300))
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img = RLImage(img_buffer, width=2.5*inch, height=2.5*inch)
        
        class_color = "#059669" if results['class_id'] == 0 else "#B45309"
        
        results_text = f"""
        <b>Class:</b> <font color='{class_color}'>{results['class_name']}</font><br/>
        <b>Class ID:</b> {results['class_id']}<br/>
        <b>Confidence:</b> {results['confidence']:.2%}<br/>
        <b>Raw Score:</b> {results['raw_score']:.4f}<br/>
        <b>Timestamp:</b> {results['timestamp']}
        """
        
        table_data = [[img, Paragraph(results_text, self.normal_style)]]
        table = Table(table_data, colWidths=[3*inch, 4*inch])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ('LEFTPADDING', (1, 0), (1, 0), 20),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def save(self):
        self.doc.build(self.story)

# Generate PDF for single prediction
def generate_single_prediction_pdf(image, results):
    """Generate PDF report for a single prediction"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        pdf = PDFReportGenerator(temp_filename)
        
        # Add logo at the beginning
        pdf.add_logo()
        
        pdf.add_title("Tranche Payment Classification Report")
        pdf.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        pdf.add_spacer(0.3)
        
        pdf.add_heading("Prediction Results")
        pdf.add_prediction_result(image, results)
        
        pdf.add_heading("Model Information")
        model_info = [
            ['Model Type', 'TensorFlow Lite (FP16)'],
            ['Input Size', f"{st.session_state.input_shape[1]}x{st.session_state.input_shape[2]} pixels"],
            ['Format', '.tflite'],
            ['Framework', f'TensorFlow {tf.__version__}']
        ]
        pdf.add_table(model_info)
        
        pdf.add_spacer(0.5)
        pdf.add_paragraph("This report was generated automatically by the Tranche Payment AI Model Classifier")
        pdf.add_paragraph("© 2026 RingimTech")
        
        pdf.save()
        time.sleep(0.5)
        
        with open(temp_filename, 'rb') as f:
            pdf_bytes = f.read()
        
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        try:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except:
            pass
        return None

# Generate PDF for batch predictions
def generate_batch_prediction_pdf(results_list):
    """Generate PDF report for multiple predictions"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        pdf = PDFReportGenerator(temp_filename)
        
        # Add logo at the beginning
        pdf.add_logo()
        
        pdf.add_title("Tranche Payment Batch Classification Report")
        pdf.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        pdf.add_paragraph(f"Total Documents Processed: {len(results_list)}")
        pdf.add_spacer(0.3)
        
        # Summary statistics by class
        pdf.add_heading("Summary Statistics")
        class_0_count = sum(1 for r in results_list if r['class_id'] == 0)
        class_1_count = sum(1 for r in results_list if r['class_id'] == 1)
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Documents', str(len(results_list))],
            ['Final Tranche (Class 0)', str(class_0_count)],
            ['Second Tranche (Class 1)', str(class_1_count)],
            ['Class 0 Percentage', f"{(class_0_count/len(results_list))*100:.1f}%"],
            ['Class 1 Percentage', f"{(class_1_count/len(results_list))*100:.1f}%"]
        ]
        pdf.add_table(summary_data)
        pdf.add_spacer(0.3)
        
        # Individual results
        pdf.add_heading("Individual Results")
        for i, result in enumerate(results_list, 1):
            pdf.add_paragraph(f"<b>Document {i}: {result['filename']}</b>")
            pdf.add_prediction_result(result['image'], result)
        
        pdf.add_spacer(0.5)
        pdf.add_paragraph("This report was generated automatically by the Tranche Payment AI Model Classifier")
        pdf.add_paragraph("© 2026 RingimTech")
        
        pdf.save()
        time.sleep(0.5)
        
        with open(temp_filename, 'rb') as f:
            pdf_bytes = f.read()
        
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error generating batch PDF: {str(e)}")
        try:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except:
            pass
        return None

# Sidebar
with st.sidebar:
    # Add logo to sidebar
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    st.markdown("### 📁 Model Selection")
    
    # Load model button
    if st.button("🔄 Load TFLite Model", use_container_width=True):
        with st.spinner("Loading TensorFlow Lite model..."):
            st.session_state.model = load_tranche_model()
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
                st.success("✅ TFLite model loaded successfully!")
                st.info(f"Input shape: {st.session_state.input_shape}")
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown(f"""
        <div class='success-box'>
            ✅ TFLite model ready<br>
            Input: {st.session_state.input_shape[1]}x{st.session_state.input_shape[2]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
            ⚠️ Click 'Load TFLite Model' to start
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class information
    st.markdown("### 📋 Class Information")
    st.info("""
    **Class 0:** Final_Tranche_Payment  
    **Class 1:** Second_Tranche_Payment
    """)

# Main content with logo at the top
logo_base64 = get_logo_base64()
if logo_base64:
    st.markdown(f"""
    <div class='logo-container'>
        <img src='data:image/png;base64,{logo_base64}' class='logo-image'>
        <h1 class='main-header'>Tranche Payment Document Classifier</h1>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='logo-container'>
        <h1 class='main-header'>Tranche Payment Document Classifier</h1>
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Single Prediction", "📊 Batch Prediction", "📈 Model Performance"])

# Tab 1: Single Prediction
with tab1:
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the TFLite model from the sidebar to start making predictions.")
    else:
        st.markdown("<h2 class='sub-header'>Single Document Classification</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                key="single_uploader"
            )
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Document", width=IMAGE_WIDTH)
        
        with col2:
            if uploaded_file is not None:
                st.markdown("### 🔍 Prediction Results")
                
                with st.spinner("Processing..."):
                    img_array, original_img = preprocess_image(
                        uploaded_file, 
                        target_size=(st.session_state.input_shape[1], st.session_state.input_shape[2])
                    )
                    
                    if img_array is not None:
                        predicted_class, confidence, raw_pred = predict_with_tflite(
                            st.session_state.model, 
                            img_array
                        )
                        
                        if predicted_class is not None:
                            pred_class_name = st.session_state.class_names[predicted_class]
                            class_color = "#059669" if predicted_class == 0 else "#B45309"
                            
                            # Display metrics
                            col2_1, col2_2 = st.columns(2)
                            with col2_1:
                                st.metric("Predicted Class", pred_class_name)
                            with col2_2:
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Result box
                            st.markdown(f"""
                            <div class='prediction-box'>
                                <h3 style='color: {class_color}; margin-top: 0;'>
                                    Class {predicted_class}: {pred_class_name}
                                </h3>
                                <p>Raw Score: {raw_pred:.4f}</p>
                                <p>Threshold: 0.5</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Prepare results for PDF
                            results = {
                                'class_name': pred_class_name,
                                'class_id': predicted_class,
                                'confidence': confidence,
                                'raw_score': raw_pred,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # Store in history
                            results_with_image = results.copy()
                            results_with_image['image'] = original_img
                            st.session_state.prediction_history.append(results_with_image)
                            
                            # PDF Generation
                            st.markdown("### 🖨️ Print Report")
                            
                            if st.button("📄 Generate PDF Report", use_container_width=True):
                                with st.spinner("Generating PDF..."):
                                    pdf_bytes = generate_single_prediction_pdf(original_img, results)
                                    
                                    if pdf_bytes:
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_bytes,
                                            file_name=f"tranche_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("Failed to generate PDF. Please try again.")

# Tab 2: Batch Prediction
with tab2:
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the TFLite model from the sidebar to make batch predictions.")
    else:
        st.markdown("<h2 class='sub-header'>Batch Document Classification</h2>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if uploaded_files:
            st.markdown(f"### 📊 Processing {len(uploaded_files)} Documents")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                img_array, original_img = preprocess_image(
                    uploaded_file,
                    target_size=(st.session_state.input_shape[1], st.session_state.input_shape[2])
                )
                
                if img_array is not None:
                    predicted_class, confidence, raw_pred = predict_with_tflite(
                        st.session_state.model, 
                        img_array
                    )
                    
                    if predicted_class is not None:
                        result = {
                            'filename': uploaded_file.name,
                            'class_name': st.session_state.class_names[predicted_class],
                            'class_id': predicted_class,
                            'confidence': confidence,
                            'raw_score': raw_pred,
                            'image': original_img,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        results.append(result)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            if results:
                # Summary statistics by class
                class_0_count = sum(1 for r in results if r['class_id'] == 0)
                class_1_count = sum(1 for r in results if r['class_id'] == 1)
                
                st.markdown("### 📋 Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", len(results))
                
                with col2:
                    st.metric("Final Tranche (Class 0)", class_0_count,
                             f"{(class_0_count/len(results))*100:.1f}%")
                
                with col3:
                    st.metric("Second Tranche (Class 1)", class_1_count,
                             f"{(class_1_count/len(results))*100:.1f}%")
                
                # Results table
                st.markdown("### 📋 Results Table")
                df_display = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Class': r['class_name'],
                        'Class ID': r['class_id'],
                        'Confidence': f"{r['confidence']:.2%}"
                    } for r in results
                ])
                st.dataframe(df_display, use_container_width=True)
                
                # Image gallery
                st.markdown("### 🖼️ Image Gallery")
                cols_per_row = 3
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(results):
                            result = results[i + j]
                            with col:
                                st.image(result['image'], caption=result['filename'], width=GALLERY_WIDTH)
                                color = "#059669" if result['class_id'] == 0 else "#B45309"
                                st.markdown(f"""
                                <div style='text-align: center; padding: 5px; background-color: {color}20; border-radius: 5px;'>
                                    <strong style='color: {color};'>{result['class_name']}</strong><br>
                                    Class {result['class_id']} | {result['confidence']:.1%}
                                </div>
                                """, unsafe_allow_html=True)
                
                # PDF Generation
                st.markdown("### 🖨️ Print Batch Report")
                
                if st.button("📄 Generate Batch PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = generate_batch_prediction_pdf(results)
                        
                        if pdf_bytes:
                            st.download_button(
                                label="📥 Download Batch PDF",
                                data=pdf_bytes,
                                file_name=f"tranche_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate PDF. Please try again.")

# Tab 3: Model Performance
with tab3:
    st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", "93.5%", "+6.5%")
    with col2:
        st.metric("Validation Accuracy", "100%", "Peak")
    with col3:
        st.metric("Test Loss", "0.1820", "-0.1623", delta_color="inverse")
    with col4:
        st.metric("F1 Score", "0.94", "Macro avg")
    
    # Training history
    st.markdown("### 📈 Training Progress")
    
    epochs = list(range(1, 16))
    training_acc = [0.54, 0.585, 0.62, 0.755, 0.755, 0.78, 0.795, 0.83, 0.82, 0.855, 0.855, 0.89, 0.845, 0.87, 0.88]
    val_acc = [0.625, 0.745, 0.82, 0.84, 0.88, 0.88, 0.885, 0.905, 0.91, 0.915, 0.925, 0.93, 0.94, 0.94, 0.94]
    
    df_stage1 = pd.DataFrame({
        'Epoch': epochs,
        'Training': training_acc,
        'Validation': val_acc
    })
    
    fig = px.line(df_stage1, x='Epoch', y=['Training', 'Validation'],
                  title="Training Progress",
                  color_discrete_map={'Training': '#2563EB', 'Validation': '#059669'})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p>Developed with TensorFlow Lite & Streamlit | © 2026 RingimTech</p>
    <p style='font-size: 0.8rem;'>Version 5.0.0 | TFLite FP16 Model</p>
</div>
""", unsafe_allow_html=True)
# # import os
# # import json
# # from PIL import Image

# # import numpy as np
# # import tensorflow as tf
# # import streamlit as st


# # working_dir = os.path.dirname(os.path.abspath(__file__))
# # model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# # # Load the pre-trained model
# # model = tf.keras.models.load_model(model_path)

# # # loading the class names
# # class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# # # Function to Load and Preprocess the Image using Pillow
# # def load_and_preprocess_image(image_path, target_size=(224, 224)):
# #     # Load the image
# #     img = Image.open(image_path)
# #     # Resize the image
# #     img = img.resize(target_size)
# #     # Convert the image to a numpy array
# #     img_array = np.array(img)
# #     # Add batch dimension
# #     img_array = np.expand_dims(img_array, axis=0)
# #     # Scale the image values to [0, 1]
# #     img_array = img_array.astype('float32') / 255.
# #     return img_array


# # # Function to Predict the Class of an Image
# # def predict_image_class(model, image_path, class_indices):
# #     preprocessed_img = load_and_preprocess_image(image_path)
# #     predictions = model.predict(preprocessed_img)
# #     predicted_class_index = np.argmax(predictions, axis=1)[0]
# #     predicted_class_name = class_indices[str(predicted_class_index)]
# #     return predicted_class_name


# # # Streamlit App
# # st.title('Plant Disease Classifier')

# # uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# # if uploaded_image is not None:
# #     image = Image.open(uploaded_image)
# #     col1, col2 = st.columns(2)

# #     with col1:
# #         resized_img = image.resize((150, 150))
# #         st.image(resized_img)

# #     with col2:
# #         if st.button('Classify'):
# #             # Preprocess the uploaded image and predict the class
# #             prediction = predict_image_class(model, uploaded_image, class_indices)
# #             st.success(f'Prediction: {str(prediction)}')


# import os
# import json
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import streamlit as st

# # Page config
# st.set_page_config(
#     page_title="Plant Disease Detector",
#     page_icon="🌱",
#     layout="centered"
# )

# # Custom CSS
# st.markdown("""
# <style>
# .main {background-color: #f5f7fa;}
# .stButton>button {width:100%;border-radius:10px;height:45px;}
# </style>
# """, unsafe_allow_html=True)

# working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# model = tf.keras.models.load_model(model_path)

# class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# # Functions
# def load_and_preprocess_image(image, target_size=(224,224)):
#     img = Image.open(image).convert("RGB")
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array.astype('float32') / 255.
#     return img_array

# def predict_image_class(model, image, class_indices):
#     img = load_and_preprocess_image(image)
#     preds = model.predict(img)[0]
#     top3 = preds.argsort()[-3:][::-1]
#     results = [(class_indices[str(i)], preds[i]*100) for i in top3]
#     return results

# # Header
# st.title("🌱 Plant Disease Classifier")
# st.caption("Upload a leaf image and detect disease using Deep Learning")

# # Sidebar
# st.sidebar.header("Upload Image")
# uploaded_image = st.sidebar.file_uploader("Choose image", type=["jpg","jpeg","png"])

# if uploaded_image:

#     image = Image.open(uploaded_image)

#     st.subheader("Uploaded Image")
#     st.image(image, use_container_width=True)

#     if st.button("🔍 Classify Plant"):

#         with st.spinner("Analyzing..."):

#             results = predict_image_class(model, uploaded_image, class_indices)

#         st.success("Prediction Complete ✅")

#         st.subheader("🧪 Results")

#         for label, confidence in results:
#             st.progress(int(confidence))
#             st.write(f"**{label}** — {confidence:.2f}%")

#         st.markdown("---")
#         st.info(f"🌟 Most Likely: **{results[0][0]}**")

# else:
#     st.warning("Please upload an image to begin.")

# st.markdown("---")
# st.caption("Made with ❤️ using TensorFlow + Streamlit")




import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from typing import Dict, Tuple, List
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    """Load the pre-trained model with caching."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_class_indices(json_path: str) -> Dict:
    """Load class indices from JSON file with caching."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading class indices: {str(e)}")
        return {}

def load_and_preprocess_image(image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess the image for model prediction.
    
    Args:
        image: PIL Image or file path
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array
    """
    try:
        # Handle both file paths and PIL Images
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image
        img = img.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image_class(model: tf.keras.Model, image, class_indices: Dict) -> Tuple[str, float, np.ndarray]:
    """
    Predict the class of an image.
    
    Args:
        model: Trained Keras model
        image: Input image
        class_indices: Dictionary mapping indices to class names
    
    Returns:
        Tuple of (predicted_class_name, confidence, all_probabilities)
    """
    try:
        preprocessed_img = load_and_preprocess_image(image)
        
        if preprocessed_img is None:
            return None, None, None
        
        # Get predictions
        predictions = model.predict(preprocessed_img, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        
        # Get class name
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def plot_prediction_confidence(predictions: np.ndarray, class_indices: Dict, top_n: int = 5):
    """Create a bar chart showing top N prediction confidences."""
    # Get top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_classes = [class_indices.get(str(i), f"Class {i}") for i in top_indices]
    top_confidences = [predictions[i] * 100 for i in top_indices]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=top_confidences,
            y=top_classes,
            orientation='h',
            marker=dict(
                color=top_confidences,
                colorscale='Greens',
                showscale=False
            ),
            text=[f'{conf:.2f}%' for conf in top_confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Predictions',
        xaxis_title='Confidence (%)',
        yaxis_title='Disease Class',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def get_disease_info(disease_name: str) -> Dict[str, str]:
    """
    Get additional information about the disease.
    This is a placeholder - you can expand with actual disease information.
    """
    # This is a basic template - expand with real information
    info = {
        "description": "Information about this plant disease will be displayed here.",
        "symptoms": "Common symptoms include leaf discoloration, spots, and wilting.",
        "treatment": "Recommended treatments and preventive measures.",
        "severity": "Medium"
    }
    return info

def display_image_analysis(image: Image, predicted_class: str, confidence: float, predictions: np.ndarray, class_indices: Dict):
    """Display comprehensive image analysis results."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        
        # Display image properties
        with st.expander("Image Properties"):
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'N/A'}")
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        # Main prediction
        if confidence >= 80:
            status_color = "🟢"
            status_text = "High Confidence"
        elif confidence >= 60:
            status_color = "🟡"
            status_text = "Medium Confidence"
        else:
            status_color = "🔴"
            status_text = "Low Confidence"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #4CAF50; margin: 0;">{predicted_class}</h2>
            <p style="font-size: 24px; margin: 10px 0;">{status_color} {confidence:.2f}% {status_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence chart
        st.plotly_chart(plot_prediction_confidence(predictions, class_indices), use_container_width=True)
    
    # Disease information (if applicable)
    if "healthy" not in predicted_class.lower():
        st.subheader("📋 Disease Information")
        disease_info = get_disease_info(predicted_class)
        
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Severity", disease_info["severity"])
        with info_cols[1]:
            st.metric("Confidence", f"{confidence:.1f}%")
        with info_cols[2]:
            st.metric("Status", status_text)
        
        with st.expander("Detailed Information", expanded=True):
            st.write(f"**Description:** {disease_info['description']}")
            st.write(f"**Symptoms:** {disease_info['symptoms']}")
            st.write(f"**Treatment:** {disease_info['treatment']}")

def main():
    # Header
    st.title("🌱 Plant Disease Classifier")
    st.markdown("Upload an image of a plant leaf to detect diseases using deep learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("Model Information")
        working_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
        class_indices_path = f"{working_dir}/class_indices.json"
        
        if os.path.exists(model_path):
            st.success("✓ Model loaded successfully")
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            st.info(f"Model size: {model_size:.2f} MB")
        else:
            st.error("✗ Model file not found")
            st.stop()
        
        # Load model and class indices
        model = load_model(model_path)
        class_indices = load_class_indices(class_indices_path)
        
        if model is None or not class_indices:
            st.error("Failed to load model or class indices")
            st.stop()
        
        st.info(f"Total classes: {len(class_indices)}")
        
        # Advanced options
        st.subheader("Advanced Options")
        show_top_n = st.slider("Show top N predictions", 3, 10, 5)
        confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 50)
        
        # View all classes
        with st.expander("View All Classes"):
            for idx, class_name in sorted(class_indices.items(), key=lambda x: int(x[0])):
                st.text(f"{idx}: {class_name}")
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("📊 Recent Predictions")
            for i, pred in enumerate(st.session_state.prediction_history[-5:][::-1]):
                st.text(f"{i+1}. {pred['class']} ({pred['confidence']:.1f}%)")
            
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    
    # Main content area
    st.markdown("---")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    # Sample images section
    with st.expander("ℹ️ Tips for Best Results"):
        st.markdown("""
        - Use clear, well-lit images
        - Focus on the affected area of the leaf
        - Avoid blurry or low-resolution images
        - Ensure the leaf takes up most of the frame
        - Use images with plain backgrounds when possible
        """)
    
    if uploaded_image is not None:
        try:
            # Load image
            image = Image.open(uploaded_image)
            
            # Create columns for layout
            st.markdown("---")
            
            # Classify button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                classify_button = st.button("🔍 Classify Disease", type="primary", use_container_width=True)
            
            if classify_button:
                with st.spinner("Analyzing image..."):
                    # Get prediction
                    predicted_class, confidence, predictions = predict_image_class(
                        model, image, class_indices
                    )
                    
                    if predicted_class is None:
                        st.error("Failed to classify image. Please try again.")
                    else:
                        # Add to history
                        st.session_state.prediction_history.append({
                            'class': predicted_class,
                            'confidence': confidence
                        })
                        
                        # Display results
                        if confidence >= confidence_threshold:
                            display_image_analysis(
                                image, predicted_class, confidence, 
                                predictions, class_indices
                            )
                            
                            # Download results option
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                result_data = {
                                    "prediction": predicted_class,
                                    "confidence": f"{confidence:.2f}%",
                                    "timestamp": str(np.datetime64('now'))
                                }
                                st.download_button(
                                    label="📥 Download Results (JSON)",
                                    data=json.dumps(result_data, indent=2),
                                    file_name=f"prediction_{predicted_class.replace(' ', '_')}.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning(f"Confidence ({confidence:.2f}%) is below threshold ({confidence_threshold}%). Results may not be reliable.")
                            display_image_analysis(
                                image, predicted_class, confidence, 
                                predictions, class_indices
                            )
            else:
                # Show preview before classification
                st.subheader("Image Preview")
                preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
                with preview_col2:
                    st.image(image, use_container_width=True)
                    st.info("Click 'Classify Disease' button above to analyze this image")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    else:
        # Landing page content
        st.info("👆 Upload an image to get started")
        
        # Add some visual elements
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Supported Formats", "5+", help="JPG, PNG, BMP, WEBP")
        with col2:
            st.metric("Model Classes", len(class_indices))
        with col3:
            st.metric("Predictions Made", len(st.session_state.prediction_history))

if __name__ == "__main__":
    main()
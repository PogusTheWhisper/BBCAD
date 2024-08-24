import streamlit as st
import os
from PIL import Image
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

def predict(image_path, confident):
    onnx_model = YOLO("bbcad-yolov8x.onnx")
    # Run inference
    results = onnx_model(image_path, imgsz=256, conf=confident, max_det=1)
    
    for result in results:
        boxes = result.boxes
        result.save(filename="temp.jpg")
    return "temp.jpg"

# Sidebar
with st.sidebar:
    left, center, right = st.columns(3)
    with center:
        st.image('./RB_logo.png', width=75)
    
    st.title("Config")
    
    confident = st.slider(
        label='Confident', 
        min_value=0.1, 
        max_value=1.0, 
        step=0.05, 
        value=st.session_state.get('confident', 0.02)
    )

    # Save button
    if st.button('Save Config'):
        st.session_state['confident'] = confident
        st.success("Configuration saved!")
        
    st.title("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
 
    st.title("Select an Image")
    image_folder = "sample_images"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_image = st.selectbox("Choose an image from the list", [""] + image_files)
    
    process_button = st.button("Process Image")

st.title('Bone Break Classification and Detection')
st.header("Image Display")

# Image processing logic
if process_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = uploaded_file
    elif selected_image:
        image_path = os.path.join(image_folder, selected_image)
        image = Image.open(image_path)
    else:
        st.warning("Please upload an image or select one from the list.")
        st.stop()
    
    result_path = predict(image_path, confident)
    
    st.write("Prediction Results:")
    result_image = Image.open(result_path)
    st.image(result_image, caption='Processed Image', use_column_width=True, width=100)

if selected_image:
    with st.sidebar:
        image_path = os.path.join(image_folder, selected_image)
        image = Image.open(image_path)
        st.image(image, caption='Selected Image', use_column_width=True)
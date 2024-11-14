import streamlit as st
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Directory for saving uploads and compressed files
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

def reduce_image(file_name, accuracy, output_path):
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)
    pca = PCA(n_components=accuracy)
    transformed_image = pca.fit_transform(gray_image)
    reconstructed_image = pca.inverse_transform(transformed_image)
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (
        reconstructed_image.max() - reconstructed_image.min()
    )
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    io.imsave(output_path, compressed_image_uint8)

# Streamlit Neon Themed Title
st.markdown("""
    <style>
    .neon-title {
        font-size: 48px;
        color: #ff073a;
        text-shadow: 
          0 0 10px #ff073a, 
          0 0 20px #ff073a, 
          0 0 30px #ff073a,
          0 0 40px #ff073a;
        font-family: 'Courier New', Courier, monospace;
    }
    .neon-button {
        background-color: #ff073a;
        color: #fff;
        box-shadow: 0px 0px 15px 5px rgba(255,7,58,0.75);
        font-size: 18px;
    }
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    <div class="container">
        <p class="neon-title">🖼️ Image Compression with Red Neon Effects 🎇</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("Upload your image, select compression accuracy, and download the compressed image.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Image File", type=["png", "jpg", "jpeg"])
accuracy = st.selectbox("Select Compression Accuracy", [0.8, 0.9, 0.95, 0.99], index=2)

if uploaded_file is not None:
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())
    compressed_filename = f"compressed_{uploaded_file.name}"
    compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
    reduce_image(image_path, accuracy, compressed_path)
    st.success("✅ Image compressed successfully!")
    st.image(compressed_path, caption="Compressed Image", use_column_width=True)

    # Download button with neon styling
    with open(compressed_path, "rb") as file:
        st.download_button(
            label="📥 Download Compressed Image",
            data=file,
            file_name=compressed_filename,
            mime="image/jpeg",
            key="neon-download"
        )

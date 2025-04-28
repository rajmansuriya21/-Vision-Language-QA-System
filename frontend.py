import streamlit as st
import requests
import json
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = "http://localhost:8000/api"
API_KEY = os.getenv("API_KEY")

st.set_page_config(
    page_title="Vision-Language QA System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Vision-Language QA System")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["blip2", "ofa", "llava"],
        index=0
    )
    
    # Answer style
    style = st.selectbox(
        "Answer Style",
        ["detailed", "casual", "concise"],
        index=0
    )
    
    # Explanation toggle
    explain = st.checkbox("Explain Answer", value=False)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Question input
    question = st.text_input("Enter your question about the image")

with col2:
    st.header("Output")
    
    if st.button("Get Answer") and uploaded_file is not None and question:
        try:
            # Prepare the request
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {
                "question": question,
                "model_name": model,
                "explain": explain,
                "style": style
            }
            headers = {"X-API-Key": API_KEY}
            
            # Make the request
            response = requests.post(
                f"{API_URL}/ask",
                files=files,
                params=params,
                headers=headers,
                stream=True
            )
            
            if response.status_code == 200:
                # Create a placeholder for streaming output
                answer_placeholder = st.empty()
                full_answer = ""
                
                # Stream the response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            token = line[6:]
                            full_answer += token
                            answer_placeholder.markdown(full_answer)
            else:
                st.error(f"Error: {response.text}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Upload an image and enter a question to get started!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI, Streamlit, and HuggingFace Transformers") 
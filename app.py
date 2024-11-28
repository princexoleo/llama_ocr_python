import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
from logger_config import logger

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        logger.debug("Successfully encoded image to base64")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

def extract_text_from_image(image, document_type):
    """Send image to Ollama API and get the extracted information"""
    logger.info(f"Starting text extraction for document type: {document_type}")
    
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        logger.debug("Image encoded successfully")
        
        # Prepare the prompt based on document type
        prompts = {
            "NID": "This is a Bangladeshi National ID Card. Please extract and list all important information in both Bengali and English including: \n"
                   "1. Name (both Bengali and English)\n"
                   "2. Father's Name (Bengali Only )\n"
                   "3. Mother's Name (Bengali Only)\n" 
                   "4. Date of Birth\n"
                   "5. ID Number\n"
                   "Please format the output clearly, showing both Bengali and English versions where available. "
                   "For Bengali text, please preserve the exact Bengali characters.",
            "Driving License": "This is a Driving License. Please extract and list all important information like Name, License Number, Issue Date, Expiry Date, etc.",
            "Passport": "This is a Passport. Please extract and list all important information like Name, Passport Number, Date of Birth, Expiry Date, etc."
        }
        
        # Prepare the request to Ollama API
        api_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2-vision",
            "prompt": prompts[document_type],
            "stream": False,
            "images": [base64_image]
        }

        logger.debug("Sending request to Ollama API")
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Successfully received response from Ollama API")
            return result.get('response', 'No text extracted')
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logger.error(f"API request failed: {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def main():
    logger.info("Starting Document Information Extractor application")
    
    st.set_page_config(page_title="Document Information Extractor", page_icon="📄")
    
    st.title("Document Information Extractor using Vision LLM")
    st.write("Upload an image of your document (NID, Driving License, or Passport) to extract information.")
    
    # Add info about bilingual support
    if st.sidebar.checkbox("Show Information"):
        st.sidebar.info("""
        ### Bilingual Support
        For Bangladeshi NID cards, the system extracts information in both:
        - বাংলা (Bengali)
        - English
        
        Make sure the image is clear and both languages are visible.
        """)
        logger.debug("Information sidebar displayed")

    # Document type selection
    document_type = st.selectbox(
        "Select Document Type",
        ["NID", "Driving License", "Passport"]
    )
    logger.debug(f"Selected document type: {document_type}")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        logger.info(f"File uploaded: {uploaded_file.name}")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_column_width=True)
        logger.debug("Image displayed in UI")

        # Extract button
        if st.button("Extract Information"):
            logger.info("Starting information extraction process")
            with st.spinner("Extracting information..."):
                # Process the image and get results
                result = extract_text_from_image(image, document_type)
                
                # Display results
                st.subheader("Extracted Information:")
                st.write(result)
                logger.info("Information extraction completed and displayed")

if __name__ == "__main__":
    main()

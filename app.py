import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
from logger_config import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from performance_monitor import measure_time, performance_monitor
import time
from queue import Queue
import threading
from functools import partial
import hashlib
import re

# Global thread pool for handling requests
executor = ThreadPoolExecutor(max_workers=4)
# Request queue for rate limiting
request_queue = Queue(maxsize=100)
# Rate limiting settings
MAX_REQUESTS_PER_SECOND = 10
REQUEST_WINDOW = 1.0  # seconds

class RateLimiter:
    def __init__(self, max_requests, window):
        self.max_requests = max_requests
        self.window = window
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND, REQUEST_WINDOW)

def format_driving_license_response(text):
    """Format the driving license response to ensure consistent JSON output"""
    try:
        # If response is already valid JSON, parse it
        try:
            data = json.loads(text)
            # Ensure all required fields exist
            required_fields = ["name", "date_of_birth", "father_husband", 
                             "license_number", "issue_date", "expiry_date", 
                             "issuing_authority"]
            for field in required_fields:
                if field not in data:
                    data[field] = None
            return json.dumps(data, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract information using patterns
            data = {
                "name": None,
                "date_of_birth": None,
                "father_husband": None,
                "license_number": None,
                "issue_date": None,
                "expiry_date": None,
                "issuing_authority": None
            }
            
            # Extract information using more flexible patterns
            patterns = {
                "name": [
                    r'["\']?name["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'Name\s*:?\s*([^\n,]+)',
                    r'(?:^|\n)\s*([A-Z][A-Za-z\s.]+?)(?=\s*(?:S/O|D/O|W/O|Father|Date|License|$))'
                ],
                "date_of_birth": [
                    r'["\']?date_of_birth["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'Date of Birth\s*:?\s*([^\n,]+)',
                    r'(?:DOB|Birth)\s*:?\s*([^\n,]+)'
                ],
                "father_husband": [
                    r'["\']?father_husband["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'(?:S/O|D/O|W/O|Father|Husband)\s*:?\s*([^\n,]+)',
                    r'(?:Father|Husband).s Name\s*:?\s*([^\n,]+)'
                ],
                "license_number": [
                    r'["\']?license_number["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'License\s*(?:No\.|Number)\s*:?\s*([A-Z0-9]+)',
                    r'(?:^|\n)\s*([A-Z]{2}\d+(?:CL|DL)\d+)\s*(?:$|\n)'
                ],
                "issue_date": [
                    r'["\']?issue_date["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'Issue\s*Date\s*:?\s*([^\n,]+)',
                    r'Date of Issue\s*:?\s*([^\n,]+)'
                ],
                "expiry_date": [
                    r'["\']?expiry_date["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'(?:Expiry|Valid Until)\s*Date\s*:?\s*([^\n,]+)',
                    r'Valid\s*(?:Till|Until)\s*:?\s*([^\n,]+)'
                ],
                "issuing_authority": [
                    r'["\']?issuing_authority["\']?\s*:?\s*["\']?(.*?)["\']?(?=\s*[,}\n])',
                    r'(?:Issuing Authority|Authority|Issued By)\s*:?\s*([^\n,]+)',
                    r'(?:BRTA|Bangladesh Road Transport Authority)[^\n,]*'
                ]
            }
            
            # Try each pattern for each field
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value = match.group(1).strip().strip('"\'').strip() if match.groups() else match.group(0).strip()
                        if value and value.lower() not in ['null', 'none', '']:
                            data[field] = value
                            break
            
            return json.dumps(data, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error formatting driving license response: {str(e)}")
        return text

def format_nid_response(text, validated_nid=None):
    """Format the NID response to ensure consistent JSON output"""
    try:
        # Initialize default structure
        data = {
            "name_bn": None,
            "name_en": None,
            "father_name": None,
            "mother_name": None,
            "date_of_birth": None,
            "id_number": validated_nid
        }

        # Extract information using patterns
        patterns = {
            "name_bn": [
                r'নাম:\s*(.+?)(?=\n|$)',
                r'নাম\s*:\s*(.+?)(?=\n|$)'
            ],
            "name_en": [
                r'Name:\s*(.+?)(?=\n|$)',
                r'Name\s*:\s*(.+?)(?=\n|$)'
            ],
            "father_name": [
                r'পিতা:\s*(.+?)(?=\n|$)',
                r'পিতা\s*:\s*(.+?)(?=\n|$)'
            ],
            "mother_name": [
                r'মাতা:\s*(.+?)(?=\n|$)',
                r'মাতা\s*:\s*(.+?)(?=\n|$)'
            ],
            "date_of_birth": [
                r'Date of Birth:?\s*(.+?)(?=\n|$)',
                r'Date of Birth\s*:\s*(.+?)(?=\n|$)',
                r'Birth\s*:\s*(.+?)(?=\n|$)'
            ]
        }

        # Try each pattern for each field
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in ['null', 'none', '']:
                        data[field] = value
                        break

        return json.dumps(data, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error formatting NID response: {str(e)}")
        return text

@measure_time
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=85)
        logger.debug("Successfully encoded image to base64")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

@measure_time
def extract_text_from_image(image, document_type):
    """Send image to Ollama API and get the extracted information"""
    logger.info(f"Starting text extraction for document type: {document_type}")
    
    try:
        # Wait for rate limiter
        while not rate_limiter.acquire():
            time.sleep(0.1)
        
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        logger.debug("Image encoded successfully")
        
        # Prepare the prompt based on document type
        prompts = {
            "NID": "This is a Bangladeshi National ID Card. Extract the following information and return it in valid JSON format:\n\n"
                  "Required Fields:\n"
                  "1. Name in Bengali (after নাম:)\n"
                  "2. Name in English (after Name:)\n"
                  "3. Father's Name in Bengali (after পিতা:)\n"
                  "4. Mother's Name in Bengali (after মাতা:)\n"
                  "5. Date of Birth (format: DD MMM YYYY)\n"
                  "6. ID Number (17 or 10 digits, starts with birth year)\n\n"
                  "Special Instructions for ID Number:\n"
                  "- Must be exactly 17 or 10 digits\n"
                  "- Look for it after 'ID NO:'\n\n"
                  "Example Format:\n"
                  "{\n"
                  '    "name_bn": "Bengali name here",\n'
                  '    "name_en": "English name here",\n'
                  '    "father_name": "Father name in Bengali",\n'
                  '    "mother_name": "Mother name in Bengali",\n'
                  '    "date_of_birth": "DD MMM YYYY",\n'
                  '    "id_number": "17 or 10 digits"\n'
                  "}\n\n"
                  "Important:\n"
                  "1. Keep Bengali text exactly as shown\n"
                  "2. Maintain the exact field names\n"
                  "3. Use null for missing fields\n"
                  "4. Double-check the ID number accuracy",
            "Driving License": "Analyze this Driving License image and extract the following information in JSON format. "
                             "Pay special attention to these fields:\n\n"
                             "1. Look for the license holder's full name at the top of the license\n"
                             "2. Find Father's/Husband's name (usually prefixed with S/O, D/O, or W/O)\n"
                             "3. Look for 'BRTA' or 'Bangladesh Road Transport Authority' as the issuing authority\n\n"
                             "Return the information in this exact JSON format:\n"
                             "{\n"
                             '    "name": "Full Name as shown",\n'
                             '    "date_of_birth": "DD MMM YYYY",\n'
                             '    "father_husband": "Full name with S/O or W/O prefix",\n'
                             '    "license_number": "Format: XX9999999XX9999",\n'
                             '    "issue_date": "DD MMM YYYY",\n'
                             '    "expiry_date": "DD MMM YYYY",\n'
                             '    "issuing_authority": "BRTA or full authority name"\n'
                             "}\n\n"
                             "Important Instructions:\n"
                             "1. Return ONLY the JSON object, no additional text\n"
                             "2. Use null for fields that are not visible or unclear\n"
                             "3. Include the full name exactly as shown on the license\n"
                             "4. Keep the complete Father's/Husband's name including S/O, D/O, or W/O prefix\n"
                             "5. For issuing authority, prefer 'BRTA' or 'Bangladesh Road Transport Authority'\n"
                             "6. Keep exact date formats (DD MMM YYYY)\n"
                             "7. Preserve exact license number format\n"
                             "8. Use these exact field names\n"
                             "9. Do not add any additional fields",
            "Passport": "This is a Passport. Please extract and list all important information like Name, Passport Number, Date of Birth, Expiry Date. Keep consistant outputs.."
        }
        
        # Prepare the request to Ollama API
        api_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2-vision",
            "prompt": prompts[document_type],
            "stream": False,
            "images": [base64_image],
            "num_predict": 512,  # Limit token generation
            "temperature": 0.1,   # More focused responses
            "top_p": 0.95,        # Slightly higher top_p for better number recognition
            "repeat_penalty": 1.1 # Reduce repetition
        }

        logger.debug("Sending request to Ollama API")
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            extracted_text = result.get('response', 'No text extracted')
            
            # Format response based on document type
            if document_type == "NID":
                # First validate NID number
                nid_number = validate_nid_number(extracted_text)
                # Then format the entire response as JSON
                return format_nid_response(extracted_text, nid_number)
            elif document_type == "Driving License":
                return format_driving_license_response(extracted_text)
            return extracted_text
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logger.error(f"API request failed: {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

def validate_nid_number(text):
    """Extract and validate NID number from text"""
    # Look for NID number patterns
    patterns = [
        r'ID\s*NO:?\s*(\d{17})',  # Exact 17 digits after ID NO:
        r'ID\s*NO:?\s*(\d{4}\s*\d{13})',  # With possible space after year
        r'(\d{4}\d{13})',  # Just the 17 digits together
        r'(\d{4}[\s-]?\d{13})',  # With possible separator after year
        r'ID\s*NO:?.+?(\d{4})\D*?(\d{13})'  # Split year and rest with possible text between
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match.groups()) == 2:  # Split pattern
                nid = match.group(1) + match.group(2)
            else:
                nid = match.group(1)
            
            # Clean the number
            nid = ''.join(filter(str.isdigit, nid))
            
            # Validate length and year
            if len(nid) == 17 and 1900 <= int(nid[:4]) <= 2023:
                return nid
            
    return None

@measure_time
def optimize_image(image):
    """Optimize image size and format for faster processing"""
    try:
        # Resize large images while maintaining aspect ratio
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Optimize image quality
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        optimized_image = Image.open(img_byte_arr)
        
        logger.debug(f"Image optimized to size: {optimized_image.size}")
        return optimized_image
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        return image

def process_image_async(image):
    """Process image asynchronously using ThreadPoolExecutor"""
    with ThreadPoolExecutor() as executor:
        future = executor.submit(optimize_image, image)
        return future.result()

def main():
    logger.info("Starting Document Information Extractor application")
    
    st.title("Document Information Extractor using Vision LLM")
    st.write("Upload an image of your document (NID, Driving License, or Passport) to extract information.")
    
    # Display performance metrics in sidebar
    st.sidebar.title("Performance Metrics")
    if st.sidebar.checkbox("Show Performance Stats"):
        stats = performance_monitor.get_stats()
        st.sidebar.metric("Average Response Time", f"{stats['avg_response_time']:.2f}s")
        st.sidebar.metric("Requests Per Second", f"{stats['requests_per_second']:.2f}")
        st.sidebar.metric("Total Requests", stats['total_requests'])
        st.sidebar.metric("Min Response Time", f"{stats['min_response_time']:.2f}s")
        st.sidebar.metric("Max Response Time", f"{stats['max_response_time']:.2f}s")
    
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
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            logger.info(f"File uploaded: {uploaded_file.name}")
            
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Optimize image asynchronously
            with st.spinner("Optimizing image..."):
                optimized_image = process_image_async(image)
            
            # Display the image
            st.image(optimized_image, caption="Uploaded Document", use_column_width=True)
            logger.debug("Image displayed in UI")

            # Extract button
            if st.button("Extract Information", key=f"extract_{uploaded_file.name}"):
                logger.info("Starting information extraction process")
                with st.spinner("Extracting information..."):
                    start_time = time.time()
                    # Process image directly without caching
                    result = extract_text_from_image(optimized_image, document_type)
                    end_time = time.time()
                    
                    # Display results
                    st.subheader("Extracted Information:")
                    if document_type == "Driving License":
                        try:
                            # Parse the JSON to display it nicely
                            json_data = json.loads(result)
                            st.json(json_data)
                        except json.JSONDecodeError:
                            st.write(result)
                    elif document_type == "NID":
                        try:
                            # Parse the JSON to display it nicely
                            json_data = json.loads(result)
                            st.json(json_data)
                        except json.JSONDecodeError:
                            st.write(result)
                    else:
                        st.write(result)
                    
                    # Display execution time
                    execution_time = end_time - start_time
                    st.info(f"Extraction completed in {execution_time:.2f} seconds")
                    logger.info(f"Information extraction completed in {execution_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            st.error(f"Error processing the uploaded file: {str(e)}")

if __name__ == "__main__":
    main()

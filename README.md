# Building a Bilingual Document Information Extractor with Llama Vision and Streamlit

![Cover Image](https://raw.githubusercontent.com/princexoleo/llama_ocr_python/main/sample_test_images/Create%20a%20cover%20image%20that%20describe%20the%20OCR%20application%20with%20Llama%20vision%20Language%20Model%2C%20futureistic%20(1).jpeg)

This application uses Llama Vision model to extract information from identity documents like National ID Cards, Driving Licenses, and Passports.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed with llama2-vision model
3. Virtual environment (recommended)

## Installation

1. Clone this repository
2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv ocr_env
source ocr_env/Scripts/activate  # On Windows: ocr_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running with llama2-vision model:
```bash
ollama run llama2-vision
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Select the document type from the dropdown menu

4. Upload an image of your document

5. Click "Extract Information" to process the image

## Features

- Support for multiple document types:
  - National ID Card (NID)
  - Driving License
  - Passport
- Real-time information extraction
- User-friendly interface
- Local processing using Ollama

## Note

Ensure that the Ollama service is running before starting the application. The application requires the llama2-vision model to be available locally.

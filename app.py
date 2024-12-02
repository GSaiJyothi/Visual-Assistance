from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import cv2
import pyttsx3
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GooglePalm

# Configure Google Generative AI
API_KEY = "Replace with your API key"
genai.configure(api_key=API_KEY)

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 1: Real-Time Scene Understanding
def real_time_scene_understanding(image_path):
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        response = model.generate_content(img)
        return response.text
    except Exception as e:
        return f"Error in scene understanding: {e}"

# Step 2: OCR-Based Text-to-Speech
def text_to_speech(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        
        # Initialize text-to-speech engine
        engine = pyttsx3.init()
        engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')  # Example for a female voice
        engine.say(text)
        engine.runAndWait()

        return text
    except Exception as e:
        return f"Error in text-to-speech conversion: {e}"

# Main Function
if __name__ == "__main__":
    # Paths to different images
    scene_image_path = (r"C:\Users\nikhi\Downloads\dog.jpg")     # Scene understanding image
    #text_image_path = (r"C:\Users\nikhi\Downloads\quote.jpg")         # Text extraction image
    # Step 1: Real-Time Scene Understanding
    print("Performing Scene Understanding...")
    scene_description = real_time_scene_understanding(scene_image_path)
    print("Scene Description:", scene_description)

    # Display scene image
    img = Image.open(scene_image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Real-Time Scene Understanding")
    plt.show()

    # Step 2: OCR-Based Text-to-Speech
    print("\nExtracting Text and Converting to Speech...")
    extracted_text = text_to_speech(text_image_path)
    print("Extracted Text:", extracted_text)

    # Display text image
    img = Image.open(text_image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Text-to-Speech Conversion")
    plt.show()

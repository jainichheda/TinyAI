from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivymd.app import MDApp
from kivy.core.window import Window  # For setting window size on desktop
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivymd.uix.menu import MDDropdownMenu
import tkinter as tk
from tkinter import filedialog
import threading
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator
from kivy.metrics import dp
import requests
import webbrowser
import torch
import open_clip
from PIL import Image
import cv2
import wikipediaapi
import webbrowser
from ultralytics import YOLO
from google.cloud import vision
import tensorflow as tf
import numpy as np
import requests
import os
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import OneLineAvatarListItem
import open_clip
from kivy.uix.image import AsyncImage
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.dialog import MDDialog
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import AsyncImage
from kivy.graphics.texture import Texture
from pyzbar.pyzbar import decode
import pyperclip
from kivymd.uix.button import MDFlatButton
from kivymd.uix.button import MDRaisedButton
from kivy.graphics import Color, Rectangle
from pymongo import MongoClient
from datetime import datetime
import hashlib  # For password hashing (optional)
from kivy.uix.modalview import ModalView
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.errors import ServerSelectionTimeoutError
from bson import ObjectId  # Ensure user ID is stored as ObjectId
import json
import os
from kivymd.uix.list import OneLineAvatarIconListItem, IconLeftWidget
from kivy.properties import ListProperty
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.list import OneLineListItem
from kivymd.uix.dialog import MDDialog
from kivy.uix.image import Image
from kivymd.uix.card import MDCard
from kivy.uix.widget import Widget
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivy.properties import DictProperty
from kivymd.uix.button import MDIconButton
from kivymd.uix.card import MDCard
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield import MDTextField
from kivy.uix.spinner import Spinner
from kivymd.toast import toast
import re
from collections import Counter
from sklearn.cluster import KMeans
from functools import partial
from kivy.uix.image import Image as KivyImage  # Rename Kivy Image
from PIL import Image as PILImage  # Rename Pillow Image
from kivy.core.window import Window






pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Modify this path

# Load the CLIP model with correct model and dataset names
model_name = "ViT-B-32"  # Corrected model name
pretrained_dataset = "openai"  # Dataset used for pretraining

# Create model
model = open_clip.create_model(model_name, pretrained=pretrained_dataset)

# Get preprocessing function
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)

# Load tokenizer
tokenizer = open_clip.get_tokenizer(model_name)  # Use corrected model name


# Load YOLO Model
yolo_model = YOLO("yolov8n.pt")

# Wikipedia API Setup
wiki_api = wikipediaapi.Wikipedia(language="en", user_agent="TinyAI/1.0")
GOOGLE_API_KEY = "AIzaSyBZPHRgHCJrJFE0k2WpYWxt_QPQQKaFDkE"
CX_ID = "406c8a3522faf447d"

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it"
}

timestamp = datetime.strptime("2025-03-07 21:45:00.000000", "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %I:%M %p")

SESSION_FILE = "session.json"

Window.size = (350, 620)  # Comment this line when running on an actual mobile device


"""class BaseScreen(Screen):
    def go_back(self):
       
        self.manager.transition.direction = 'right'
        self.manager.current = 'main'"""

class FileSelector:
    """ Utility class for file selection to avoid code repetition. """

    @staticmethod
    def select_file(callback):
        """ Opens file dialog in the main thread using Clock.schedule_once(). """
        Clock.schedule_once(lambda dt: FileSelector._open_file_dialog(callback), 0)

    @staticmethod
    def _open_file_dialog(callback):
        """ Internal method to open Tkinter file dialog safely in the main thread. """
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("All files", "*.*")])

        if file_path:
            print(f"File selected: {file_path}")
            callback(file_path)  # Pass the file path to the callback function
        else:
            print("No file selected")

class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.switch_to_main, 10)  # 3-second delay

    def switch_to_main(self, dt):
        self.manager.current = "main"

class MainScreen(Screen):
    def search_images(self):
        """Handles search input and retrieves images based on keywords."""
        query = self.ids.search_input.text.strip()  # Get the search text

        if not query:
            self.ids.search_results.text = "Please enter a search term."
            return

        # Call an API or local database to get relevant images (replace with your logic)
        search_results = self.fetch_images_from_api(query)

        if search_results:
            self.display_search_results(search_results)
        else:
            self.ids.search_results.text = "No results found."

    def fetch_images_from_api(self, query):
        """Fetches images from Google Custom Search API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "cx": CX_ID,
            "key": GOOGLE_API_KEY,
            "searchType": "image",
            "num": 10
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            image_urls = [item["link"] for item in data.get("items", [])]
            return image_urls if image_urls else ["No images found"]

        except requests.exceptions.RequestException as e:
            print(f"Error fetching images: {e}")
            return ["Error fetching images"]

    def display_search_results(self, images):
        """Displays search results in the UI."""
        self.ids.search_results.clear_widgets()

        for img_url in images:
            img = AsyncImage(source=img_url, size_hint_y=None, height=300, allow_stretch=True, keep_ratio=True)
            img.bind(on_touch_down=lambda instance, touch, url=img_url: self.show_image_popup(url, instance, touch))
            self.ids.search_results.add_widget(img)

    def show_image_popup(self, img_url, instance, touch):
        """ Show an enlarged popup when an image is clicked. """
        if instance.collide_point(*touch.pos):  # Check if touch is on the image
            popup_content = ImagePopup()
            popup_content.ids.popup_image.source = img_url  # Set image dynamically

            self.dialog = MDDialog(
                title="Image Preview",
                type="custom",
                content_cls=popup_content,
                size_hint=(None, None),
                width="340dp",  # Increased width for better text fit
                height="450dp",  # Adjust height
                auto_dismiss=True
            )
            self.dialog.open()

    def dismiss_popup(self):
        """ Close the popup. """
        if self.dialog:
            self.dialog.dismiss()



class ObjRec(Screen):
    detected_text = StringProperty("Upload or capture an image to detect objects.")
    wikipedia_url = StringProperty("")  # Store Wikipedia URL
    image_path = StringProperty("")  # Store the selected image path

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None  # Initialize camera
        self.camera_running = False  # Track camera state
        self.latest_frame = None  # Store the last captured frame
        self.camera_error_shown = False  # Ensures error is shown only once
        self.frame_error_shown = False  # Ensures frame error is shown only once

    def start_camera(self):
        """ Starts the webcam fresh every time. """

        # Reset any existing capture
        if self.capture is not None:
            self.stop_camera()

        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            if not self.camera_error_shown:
                toast("❌ Unable to access camera")
                print("❌ Error: Camera initialization failed!")
                self.camera_error_shown = True  # Show error only once
            self.capture = None
            return

        print("📷 Camera started.")
        self.camera_running = True
        self.camera_error_shown = False  # Reset flag if camera starts correctly
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        # Show the "Close Camera" button if available
        if hasattr(self.ids, "close_camera_button"):
            self.ids.close_camera_button.opacity = 1
            self.ids.close_camera_button.disabled = False

    def stop_camera(self):
        """ Stops the camera and clears UI elements. """
        if self.capture is None or not self.capture.isOpened():
            return  # ✅ Prevents redundant stopping
        if self.capture:
            self.capture.release()  # Release camera
            self.capture = None
            self.camera_running = False  # Mark camera as stopped

        cv2.destroyAllWindows()  # Close OpenCV windows
        print("📷 Camera closed.")

        detected_image = getattr(self.ids, "camera_feed", None)
        if detected_image:
            detected_image.texture = None  # Clear the previous image

        if hasattr(self.ids, "close_camera_button"):
            self.ids.close_camera_button.opacity = 0
            self.ids.close_camera_button.disabled = True
        # Reset error flags when camera stops
        self.camera_error_shown = False
        self.frame_error_shown = False

    def update_frame(self, *args):
        """ Continuously captures frames from the camera and updates UI. """
        if self.capture is None or not self.capture.isOpened():
            if not self.camera_error_shown:  # ✅ Only print the error once
                print("❌ Camera not initialized or already closed.")
                self.camera_error_shown = True  # ✅ Set flag to prevent repeated messages
            return

        ret, frame = self.capture.read()
        if not ret:
            if not self.frame_error_shown:
                toast("❌ No frame captured. Please try again.")
                print("❌ Error: No frame captured from camera!")
                self.frame_error_shown = True  # Show error only once
            return

        self.latest_frame = frame  # Store latest frame
        self.frame_error_shown = False  # Reset frame error flag if capture is successful

        # Convert frame for display in Kivy
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

        if hasattr(self.ids, "camera_feed"):
            self.ids.camera_feed.texture = texture  # Update UI

        self.camera_error_shown = False

    def capture_image(self):
        """Captures an image and processes it."""
        if not self.camera_running:
            print("📷 Starting Camera...")
            self.start_camera()

        if self.latest_frame is None:
            if not self.frame_error_shown:
                toast("❌ No frame available for capture.")
                print("❌ Error: No frame captured!")
                self.frame_error_shown = True  # Show error only once
            return

        print("📸 Image captured & processing started!")

        self.process_image(self.latest_frame, "captured_object")

        self.stop_camera()  # Stop camera after capturing

    def on_pre_leave(self):
        """ Ensure camera is closed when leaving the screen (Back Button Pressed). """
        self.stop_camera()

    def select_image(self):
        """ Opens file dialog for selecting an image. """
        #self.manager.transition.direction = 'left'
        FileSelector.select_file(self.process_image)

    def process_image(self, image_source, file_path=None):
        """Handles captured and uploaded images."""
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            self.image_path = image_source  # ✅ Set the image path to display it
            if image is None:
                toast(f"❌ Error: Unable to load image from {image_source}!")
                return
        elif isinstance(image_source, np.ndarray):
            image = image_source
            file_path = "captured_image.jpg"
            cv2.imwrite(file_path, image)  # ✅ Save the captured image
            self.image_path = file_path  # ✅ Set the image path for UI
        else:
            toast("❌ Error: Invalid image data received!")
            return

        if hasattr(self.ids, "camera_feed"):
            self.ids.camera_feed.source = self.image_path
            self.ids.camera_feed.reload()  # Force reload to update the UI

        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.detect_objects(processed_image, file_path if file_path else "captured_image")

    def detect_objects(self, image, file_path="captured_image"):
        """ Detects objects in the given image and updates UI. """

        if image is None or not isinstance(image, np.ndarray):
            print("❌ Error: Invalid image data received for object detection!")
            toast("Invalid image data!")
            return

        print(f"✅ Processing image: {file_path}")
        results = yolo_model(image)
        print(results)

        app = MDApp.get_running_app()
        objrec_collection = app.objrec_collection
        user_id = ObjectId(app.current_user_id)


        detected_objects = []
        for result in results:
            for box in result.boxes:
                label_index = int(box.cls[0])  # Ensure index is an integer

                # Ensure 'result.names' exists and index is valid
                label = result.names[label_index] if label_index < len(result.names) else "Unknown"

                # Extract confidence score (default to 1.0 if missing)
                confidence = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 1.0

                detected_objects.append((label, confidence))  # Store as tuple

        print("DEBUG: Detected Objects List:", detected_objects)  # Debugging print

        if not detected_objects:
            self.detected_text = "No objects detected."
            self.wikipedia_url = ""  # Reset Wikipedia URL
            toast("No objects detected! Try again.")
            return

        # Ensure the first detected object has both label & confidence
        if isinstance(detected_objects[0], tuple) and len(detected_objects[0]) == 2:
            first_object, confidence_score = detected_objects[0]
        else:
            print("❌ Error: Unexpected data format:", detected_objects[0])
            return  # Exit if format is incorrect

        self.detected_text = f"Detected: {first_object} ({confidence * 100}% confidence)"

        if file_path == "captured_image" and detected_objects:
            file_path = first_object.lower().replace(" ", "_")  # Convert to a safe filename format


        self.save_object_recognition(first_object, confidence_score, file_path)


        # Fetch Wikipedia info (if landmark)
        wiki_info = self.get_wikipedia_info(first_object)
        if wiki_info:
            self.wikipedia_url = wiki_info['wikipedia_url']
        else:
            self.wikipedia_url = ""

    def detect_landmark_opencv(image_path):
        """ Detects landmarks using a pre-trained AI model (ResNet). """
        model = tf.keras.applications.ResNet50(weights="imagenet")

        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize to ResNet50 input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        predictions = model.predict(image)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0]

        return [(item[1], item[2]) for item in decoded_predictions]  # (Landmark Name, Confidence Score)

    def get_wikipedia_info(self, search_term):
        """ Fetch Wikipedia details of a detected object if it's a landmark. """
        page = wiki_api.page(search_term)
        if not page.exists():
            return None
        return {
            "title": page.title,
            "description": page.summary[:300],  # Get first 300 characters
            "wikipedia_url": page.fullurl,
        }

    def open_wikipedia(self):
        """ Opens the Wikipedia page in the browser. """
        if self.wikipedia_url:
            webbrowser.open(self.wikipedia_url)

    def save_object_recognition(self, detected_object, confidence_score, image_path):
        app = MDApp.get_running_app()
        objrec_collection = app.objrec_collection  # Assuming the database collection for object recognition is named 'objrec_collection'

        if not app.is_logged_in or not app.logged_in_user:
            print("❌ No user is logged in. Cannot save object recognition data.")
            return

        user_data = app.logged_in_user
        if "_id" not in user_data:
            print("❌ Logged-in user has no valid ID.")
            return


        user_id = user_data["_id"]
        timestamp = datetime.now()

        obj_rec_entry = {
            "user_id": ObjectId(user_id),
            "detected_object": detected_object,
            "confidence_score": confidence_score,
            "image_path": image_path,  # Path or reference to the recognized object's image
            "created_at": timestamp
        }

        result = objrec_collection.insert_one(obj_rec_entry)
        objrec_id = result.inserted_id  # Get object recognition ID

        # Global function to add history
        app.add_to_history(
            user_id=app.current_user_id,
            activity_type="object_recognition",
            description=f"Objects detected: {', '.join(detected_object)}",
            feature="object_recognition",
            reference_id=objrec_id  # Link to Object Recognition result
        )
        print("✅ Object recognition data saved successfully:", result.inserted_id)

class OCR(Screen):
    result_text = StringProperty("")

    def select_ocr_file(self, instance):
        """ Allows user to select an image for OCR (Clears previous result). """
        self.reset_text()  # Clear previous result before selecting a new image
        FileSelector.select_file(self.process_ocr_file)

    def process_ocr_file(self, file_path):
        """ Handles the selected file for OCR processing. """
        print(f"OCR Processing: {file_path}")

        # Indicate that processing is in progress
        Clock.schedule_once(lambda dt: self.set_processing_text(), 0)

        # Start OCR in a separate thread
        threading.Thread(target=self.perform_ocr, args=(file_path,), daemon=True).start()

    def set_processing_text(self):
        """ Sets the processing message while OCR is running. """
        self.result_text = "Processing image..."

    def perform_ocr(self, file_path):
        app = MDApp.get_running_app()
        ocr_results_collection = app.ocr_results_collection
        user_id = ObjectId(app.current_user_id)

        """ Performs OCR on the selected file and updates result_text safely. """
        try:
            img = PILImage.open(file_path)
            ocr_result = pytesseract.image_to_string(img).strip()

            if ocr_result.strip():
                print(f"📄 Extracted Text: {ocr_result}")

                # Save to MongoDB
                self.save_ocr_result(ocr_result, img)


            else:
                print("⚠ No text detected.")

            Clock.schedule_once(lambda dt: self.update_result(ocr_result), 0)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            Clock.schedule_once(lambda dt: self.update_result(error_message), 0)

    def update_result(self, text):
        """ Updates the OCR result text in the main thread. """
        self.result_text = text if text else "No text found in image."

    def reset_text(self):
        """ Clears the result text to allow new image selection. """
        self.result_text = ""

    def save_ocr_result(self, scanned_text, image_path, source_lang="auto"):
        """Save OCR results to MongoDB."""
        app = MDApp.get_running_app()

        if not app.is_logged_in or not app.logged_in_user:
            print("❌ No user is logged in. Cannot save translation.")
            return

        user_data = app.logged_in_user  # Get user data
        if "_id" not in user_data:
            print("❌ Logged-in user has no valid ID.")
            return

        user_id = app.logged_in_user["_id"]  # Get logged-in user's ID
        timestamp = datetime.now()

        ocr_entry = {
            "user_id": ObjectId(user_id),  # Associate OCR result with a user
            "scanned_text": scanned_text,
            "image_path": str(image_path),  # Optional: Store path to the processed image
            "source_language": source_lang,
            "created_at": timestamp
        }

        result = app.ocr_results_collection.insert_one(ocr_entry)
        ocr_id = result.inserted_id  # Get OCR result ID

        # Global function to add history
        app.add_to_history(
            user_id=app.current_user_id,
            activity_type="ocr",
            description=f"OCR performed on image. Extracted: '{scanned_text[:20]}...'",
            feature="ocr",
            reference_id=ocr_id  # Link to OCR result
        )
        print(f"✅ OCR result saved with ID: {result.inserted_id}")
        return True

class Translation(Screen):
    translated_text = StringProperty("")
    selected_language_code = StringProperty("en")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.menu = None

    def open_dropdown(self):
        # Ensure menu items are defined
        menu_items = [
            {"text": language, "viewclass": "OneLineListItem", "on_release": lambda x=language: self.set_language(x)}
            for language in LANGUAGES
        ]
        # Initialize the dropdown menu if not already initialized
        if not self.menu:
            self.menu = MDDropdownMenu(
                caller=self.ids.language_dropdown,  # The widget that triggers the dropdown
                items=menu_items,
                position="center",
                width_mult=4,  # Adjust width multiplier for the dropdown menu
            )

        self.menu.open()  # Open the dropdown menu


    def set_language(self, language):
        self.ids.language_dropdown.text = language
        self.selected_language_code = LANGUAGES[language]  # Set the corresponding language code
        print(f"Selected language: {language}, Code: {self.selected_language_code}")  # Debugging output
        self.menu.dismiss()

    def translate_text(self):
        """ Translates the input text to the selected language. """
        source_text = self.ids.source_text.text.strip()
        target_language = self.selected_language_code

        if not source_text:
            self.update_result("Enter text to translate.")
            return

        if not target_language:
            self.update_result("Select a target language.")
            return
        print(f"Translating to: {target_language}")  # Debugging output

        # Run translation in a separate thread to avoid UI freeze
        Clock.schedule_once(lambda dt: self.process_translation(source_text, target_language), 0)

    def process_translation(self, source_text, target_language):
        """ Processes translation using Google Translate API. """
        app = MDApp.get_running_app()
        translations_collection = app.translations_collection
        user_id = ObjectId(app.current_user_id)
        if translations_collection is None:
            print("❌ Database connection not available, cannot save translation.")
            return None

            # Ensure the text and language are valid
        if not source_text or not target_language:
            print("⚠️ Invalid input: text or target language is missing.")
            return None

        try:
            #translator = Translator()
            print(f"🌍 Attempting translation: '{source_text}' to {target_language}")
            translated_text = GoogleTranslator(source='auto', target=target_language).translate(source_text)

            if translated_text is None:
                print("❌ Error: Translation result is None or empty.")
                Clock.schedule_once(lambda dt: self.update_result("❌ Translation failed."), 0)
                return


            if translated_text is None:
                print("❌ Error: Translation result is None. Cannot save translation.")
                return

            if isinstance(translated_text, list):
                if len(translated_text) > 0:
                    translated_text = translated_text[0]  # Take only the first translated word
                else:
                    print("❌ Error: Translation list is empty. Cannot save translation.")
                    return
            print(f"🔍 Raw translation output: {translated_text}")

            self.save_translation(source_text, translated_text, "auto", target_language)

            if translated_text:
                Clock.schedule_once(lambda dt: self.update_result(translated_text), 0)
            else:
                print("❌ Translation result is empty.")
                Clock.schedule_once(lambda dt: self.update_result("❌ Translation failed."), 0)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            Clock.schedule_once(lambda dt: self.update_result(error_message), 0)

    def update_result(self, text):
        """ Updates the translated text in the UI. """
        self.ids.translated_text.text = text

    def clear_text(self):
        """ Clears all text fields. """
        self.ids.source_text.text = ""
        self.ids.translated_text.text = ""
        self.ids.language_dropdown.text = "Select Language"

    def save_translation(self, text, translated_text, source_lang, target_lang):
        app = MDApp.get_running_app()
        translations_collection = app.translations_collection

        if not app.is_logged_in or not app.logged_in_user:
            print("❌ No user is logged in. Cannot save translation.")
            return

        user_data = app.logged_in_user  # Get user data
        if "_id" not in user_data:
            print("❌ Logged-in user has no valid ID.")
            return

        user_id = user_data["_id"] # Get the logged-in user's ID
        timestamp = datetime.now()

        translation_entry = {
            "user_id": ObjectId(user_id),
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "created_at": timestamp
        }
        result = translations_collection.insert_one(translation_entry)
        translation_id = result.inserted_id

        # Global function to add history
        app.add_to_history(
            user_id=app.current_user_id,
            activity_type="translate",
            description=f"Translated '{text}' to {translated_text}",
            feature="translate",
            reference_id=translation_id  # Link to translation
        )

class QRScanner(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.qr_detected = False
        self.qr_code = ""

    def upload_qr(self):
        """ Opens file dialog for selecting a QR code image. """
        FileSelector.select_file(self.process_selected_image)


    def process_selected_image(self, file_path):
        """ Processes the selected QR code image. """
        if file_path:
            print(f"Processing file: {file_path}")
            img = cv2.imread(file_path)
            if img is None:
                self.ids.qr_text.text = "Error: Invalid image file."
                return
            qr_code = self.process_qr_code(img)
            if qr_code:  # Show the dialog if QR code is detected
                self.show_qr_popup(qr_code)

    def start_camera(self):
        """ Starts the webcam for real-time QR scanning. """
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            self.ids.qr_text.text = "Error: Unable to access camera."
            self.capture = None
            return

        print("Camera opened successfully.")
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        if hasattr(self.ids, "close_camera_button"):
            self.ids.close_camera_button.opacity = 1
            self.ids.close_camera_button.disabled = False

    def stop_camera(self):
        """ Stops the camera and releases resources. """
        if self.capture:
            self.capture.release()  # Release camera
            self.capture = None  # Reset capture object

        cv2.destroyAllWindows()  # Close OpenCV windows
        self.qr_detected = False  # Reset detection flag
        self.camera_error_shown = False  # Reset error flag

        print("📷 Camera closed successfully.")

        if hasattr(self.ids, "qr_image"):
            self.ids.qr_image.texture = None  # Clear the camera preview

            # Hide the "Close Camera" button when the camera stops
        if hasattr(self.ids, "close_camera_button"):
            self.ids.close_camera_button.opacity = 0
            self.ids.close_camera_button.disabled = True

    def update_frame(self, *args):
        """ Continuously scans QR codes using the webcam and updates the UI. """
        if self.capture is None or not self.capture.isOpened():  # Check if camera is available
            if not hasattr(self, "camera_error_shown") or not self.camera_error_shown:
                print("❌ Camera not initialized or already closed.")
                self.camera_error_shown = True  # Set flag to prevent repeated messages
            return

        ret, frame = self.capture.read()
        if not ret:
            return  # Skip frame if capture fails

        if self.qr_detected:
            return  # Skip frame updates if QR code is already detected

        qr_code = self.process_qr_code(frame)  # Process QR code

        if qr_code:  # If QR is found, show the popup or switch screen
            self.qr_detected = True  # Mark QR as detected
            self.stop_camera()  # Stop camera instead of direct release
            self.show_qr_popup(qr_code)  # Show QR popup
            # Show the URL in a popup dialog
            self.show_qr_popup(qr_code)

            return  # Stop updating frames after QR detection

        # Update UI with webcam feed if no QR detected
        if hasattr(self.ids, "qr_image") and self.ids.qr_image is not None:
            frame = cv2.flip(frame, 0)  # Flip frame if needed
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.qr_image.texture = texture

    def on_leave(self, *args):
        """ Called when the user navigates away from this screen. """
        self.stop_camera()

    def process_qr_code(self, frame):
        app = MDApp.get_running_app()
        """ Detects QR codes and returns the extracted link. """
        decoded_objects = decode(frame)
        if decoded_objects:
            qr_text = decoded_objects[0].data.decode("utf-8")
            self.save_qr_code_scan(qr_text)
            self.ids.qr_text.text = f"QR Code: {qr_text}"
            print(f"QR Code Detected: {qr_text}")
            return qr_text  # Return the QR code text (URL)
        else:
            self.ids.qr_text.text = "No QR code detected."
            return None

    def show_qr_popup(self, qr_code):
        app = MDApp.get_running_app()
        """ Display a popup with buttons to open or copy the URL. """
        self.dialog = MDDialog(
            title="QR Code Detected",
            text="Would you like to open the link or copy it?",
            size_hint=(0.8, 0.4),
            buttons=[
                MDRaisedButton(
                    text="Search",
                    on_release=lambda x: self.open_browser(qr_code)
                ),
                MDRaisedButton(
                    text="Copy URL",
                    on_release=lambda x: app.copy_text(qr_code)
                ),
                MDFlatButton(
                    text="X",  # Close button
                    size_hint=(None, None),
                    size=("40dp", "40dp"),
                    pos_hint={"left": 0, "top": 1},  # Position it in the upper left
                    on_release=self.close_dialog,  # Call the close_dialog method
                ),
            ]
        )
        self.dialog.open()
    def open_browser(self, qr_code):
        """ Opens the URL in the default web browser. """
        webbrowser.open(qr_code)
        self.close_dialog()


    def close_dialog(self, *args):
        """ Close the dialog. """
        if self.dialog:
            self.dialog.dismiss()

    def save_qr_code_scan(self, scanned_data):
        """ Saves QR code scan details to the database. """
        app = MDApp.get_running_app()
        qr_collection = app.qrcode_collection  # Reference to QR code database collection

        if not app.is_logged_in or not app.logged_in_user:
            print("❌ No user is logged in. Cannot save QR scan data.")
            return

        user_data = app.logged_in_user
        if "_id" not in user_data:
            print("❌ Logged-in user has no valid ID.")
            return

        user_id = user_data["_id"]
        timestamp = datetime.now()

        qr_scan_entry = {
            "user_id": ObjectId(user_id),
            "scanned_data": scanned_data,  # Store the scanned QR code content
            "scan_time": timestamp,
            "scan_type": "QR Code"  # Could be "QR Code" or "Barcode"
        }

        result = qr_collection.insert_one(qr_scan_entry)
        scan_id = result.inserted_id

        app.add_to_history(
            user_id=app.current_user_id,
            activity_type="QR Code Scan",
            description="Scanned a QR Code successfully",
            feature="qr_code",
            reference_id=scan_id  # Store reference to this QR scan
        )

class ColorExtraction(Screen):
    image_path = None
    def select_image(self):
        """ Opens file dialog for selecting a QR code image. """
        FileSelector.select_file(self.process_selected_image)


    def process_selected_image(self, file_path):
        """ Processes the selected QR code image. """
        if isinstance(file_path, list) and len(file_path) > 0:
            self.image_path = os.path.normpath(file_path[0])  # If it's a list, take the first item
        elif isinstance(file_path, str) and os.path.exists(file_path):
            self.image_path = os.path.normpath(file_path)  # If it's a string, use it directly
        else:
            toast("Error: No image selected or invalid file path.")
            return
        self.ids.image_display.source = self.image_path

    def extract_colors(self):
        app = MDApp.get_running_app()
        if not self.image_path:
            toast("No image selected!")
            return

        extracted_colors = self.extract_palette_with_percentage(self.image_path)

        color_layout = self.ids.color_palette
        color_layout.clear_widgets()
        sorted_colors = sorted(extracted_colors.items(), key=lambda item: item[1], reverse=True)
        for color, percentage in sorted_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            color_button = MDRaisedButton(
                text=f"{percentage:.1f}%",
                md_bg_color=[color[0] / 255, color[1] / 255, color[2] / 255, 1],
                on_release=lambda btn, c=hex_color: app.copy_text(c)
            )
            color_layout.add_widget(color_button)
        print("Extracted colors (before saving):", extracted_colors)
        print("Type of extracted_colors before function call:", type(extracted_colors))
        self.save_extracted_colors(extracted_colors)

        '''color_box = MDBoxLayout(
            size_hint=(None, None),
            size=(50, 50),
            md_bg_color=[color[0] / 255, color[1] / 255, color[2] / 255, 1],
            on_touch_down=lambda instance, touch: app.copy_text(hex_color)
        )

        color_layout.add_widget(color_box)'''

    def extract_palette_with_percentage(self, image_path, num_colors=5, white_threshold=210):
        if not os.path.exists(image_path):  # Check if file exists
            toast("Error: Image file not found!")
            return {}
        image = cv2.imread(image_path)
        if image is None:
            toast("Error: Could not read image.")
            return {}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(image)

        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_count = sum(label_counts.values())

        extracted_colors = {
            tuple(kmeans.cluster_centers_[i].astype(int)): (label_counts[i] / total_count) * 100
            for i in label_counts
        }

        # **Filter out white & near-white colors**
        filtered_colors = {
            color: percentage
            for color, percentage in extracted_colors.items()
            if not (color[0] > white_threshold and color[1] > white_threshold and color[2] > white_threshold)
               or (color[0] > 200 and color[1] > 200 and color[2] < 150)
        }

        return filtered_colors

    def save_extracted_colors(self, extracted_colors):
        """ Saves extracted color data to the database. """
        print("Extracted colors inside save_extracted_colors:", extracted_colors)
        print("Type of extracted_colors inside function:", type(extracted_colors))

        # Ensure extracted_colors is a dictionary
        if not isinstance(extracted_colors, dict):
            print("❌ Error: extracted_colors should be a dictionary but got", type(extracted_colors))
            return

        app = MDApp.get_running_app()
        color_collection = app.color_extraction_collection  # Reference to the color database collection

        if not app.is_logged_in or not app.logged_in_user:
            print("❌ No user is logged in. Cannot save color extraction data.")
            return

        user_data = app.logged_in_user
        if "_id" not in user_data:
            print("❌ Logged-in user has no valid ID.")
            return

        user_id = user_data["_id"]
        timestamp = datetime.now()

        # ✅ Convert extracted_colors keys from np.int64 to normal int
        processed_colors = {
            (int(r), int(g), int(b)): f"{percentage:.2f}%"  # Convert np.int64 to standard int
            for (r, g, b), percentage in extracted_colors.items()
        }

        # ✅ Store as a list of dictionaries in MongoDB
        colors_list = [
            {
                "hex_color": "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2]),
                "rgb": {"r": color[0], "g": color[1], "b": color[2]},
                "percentage": percentage
            }
            for color, percentage in processed_colors.items()
        ]

        # ✅ MongoDB Document Structure
        color_extraction_entry = {
            "user_id": ObjectId(user_id),
            "extracted_colors": colors_list,
            "extraction_time": timestamp
        }

        # ✅ Insert into MongoDB
        result = color_collection.insert_one(color_extraction_entry)
        extraction_id = result.inserted_id  # Get the inserted document's ID

        print("✅ Extracted color data saved successfully:", extraction_id)

        # **Log the extraction in history**
        app.add_to_history(
            user_id= app.current_user_id,
            activity_type="Color Extraction",
            description="Extracted colors from an image",
            feature="Color_Detection",
            reference_id=extraction_id  # Store reference to this extraction
        )
        print("📜 Extraction history logged successfully!")

class ImagePopup(BoxLayout):
    pass

class ClickableImage(ButtonBehavior, AsyncImage):
    def on_press(self):
        if self.source:  # Open the image source URL
            webbrowser.open(self.source)

class ProfileScreen(Screen):
    username = StringProperty("User Name")
    user_email = StringProperty("user@example.com")
    @staticmethod
    def get_user_data(user_id):
        app = MDApp.get_running_app()
        try:
            if not isinstance(user_id, ObjectId):
                user_id = ObjectId(user_id)
            # Fetch the user data based on user_id
            user = app.users_collection.find_one({"_id": user_id})
            # Check if user is logged in and data exists
            if user:
                username = user.get("username", "User Name")
                email = user.get("email", "user@example.com")
                return username, email
            else:
                return "User Name", "user@example.com"
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return "User Name", "user@example.com"

    def update_user_info(self, username, email):
        app = MDApp.get_running_app()
        if hasattr(app, "current_user_id") and app.current_user_id:
            username, email = self.get_user_data(app.current_user_id)
            print(f"🔄 Updating profile screen with username: {username} and email: {email}")
            Clock.schedule_once(lambda dt: self.update_labels(username, email))

        else:
            self.ids.username.text = "User Name"
            self.ids.user_email.text = "user@example.com"

    def update_labels(self, username, email):
        print("Updating labels...")
        print(f"Username label: {self.ids.username}")
        print(f"Email label: {self.ids.user_email}")
        self.username = str(username)
        self.user_email = str(email)
        print("Labels updated!")

    def check_login(self):
        app = MDApp.get_running_app()
        """Check if user is logged in before accessing History."""
        if app.is_logged_in:
            app.root.current = "history"
        else:
            toast("Please log in to access History")
            app.root.current = "login"

class LoginScreen(Screen):

    def validate_credentials(self, username, password):
        """ Check user credentials in MongoDB """
        app = MDApp.get_running_app()
        user = app.users_collection.find_one({"username": username})

        if user:
            # If passwords are hashed, compare hashes
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            if user["password"] == hashed_password:
                print("✅ Login successful!")
                return True
            else:
                print("❌ Incorrect password!")
                app.show_popup("Invalid Password")
                return False
        else:
            print("❌ User not found!")
            app.show_popup("User not found")
            return False

    def login_success(self, username, password):
        """Mark user as logged in and redirect to history."""
        app = MDApp.get_running_app()
        user = app.users_collection.find_one({"username": username})

        if self.validate_credentials(username, password):
            app.is_logged_in = True
            app.logged_in_user = user  # Store full user data
            app.current_user_id = str(user["_id"])  # Store user ID as string
            print(f"✅ Logged in as: {app.current_user_id}")  # Debugging
            app.root.current = "history"

            with open(SESSION_FILE, "w") as file:
                json.dump({"user_id": str(user["_id"])}, file)
            print(f"✅ Login successful for {username}")
            return True
        else:
            print("❌ Invalid username or password")
            return False

    def load_session(app):
        """Load user session from file."""
        app = MDApp.get_running_app()

        if not os.path.exists(SESSION_FILE):
            print("🔹 No saved session found.")
            return

        try:
            with open(SESSION_FILE, "r") as file:
                session_data = json.load(file)
                user_id = session_data.get("user_id")

                if user_id:
                    user = app.users_collection.find_one({"_id": ObjectId(user_id)})
                    if user:
                        app.is_logged_in = True
                        app.logged_in_user = user
                        app.current_user_id = str(user["_id"])  # Store user ID
                        print(f"✅ Session restored for {user['username']}")
                    else:
                        print("❌ Session invalid: User not found")
                        app.is_logged_in = False
                        app.current_user_id = None
                else:
                    print("❌ Session file corrupted")

        except Exception as e:
            print(f"❌ Error loading session: {e}")
            app.is_logged_in = False
            app.current_user_id = None

class SignupScreen(Screen):
    def validate_signup(self):
        """Check if all signup fields are filled before proceeding."""
        app = MDApp.get_running_app()
        screen = app.root.get_screen("signup")

        name = screen.ids.full_name.text.strip()
        username = screen.ids.username.text.strip()
        email = screen.ids.email.text.strip()
        phone = screen.ids.phone.text.strip()
        password = screen.ids.password.text.strip()
        confirm_password = screen.ids.confirm_password.text.strip()

        error_messages = []  # List to store errors

        if not name:
            error_messages.append("Full Name is required!")

        if not username:
            error_messages.append("Username is required!")

        if not email:
            error_messages.append("Email is required!")

        if not phone:
            error_messages.append("Phone Number is required!")

        if not password:
            error_messages.append("Password is required!")

        if not confirm_password:
            error_messages.append("Confirm Password is required!")

        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if email and not re.match(email_regex, email):
            error_messages.append("Invalid Email Format!")

        # Validate phone number (10-digit numeric)
        if phone and (not phone.isdigit() or len(phone) != 10):
            error_messages.append("Phone Number must be 10 digits!")

        if password and confirm_password and password != confirm_password:
            error_messages.append("Passwords do not match!")

        if error_messages:
            for error in error_messages:
                toast(error)  # Show each error as a toast
            return False

        return True

    def signup(self, username, password, full_name, email, phone):
        app = MDApp.get_running_app()
        """ Register a new user if they don’t exist """
        if not self.validate_signup():  # If validation fails, stop execution
            return False

        if app.users_collection.find_one({"username": username}):
            app.show_popup("❌ Username already exists!")
            return False

        # Hash the password before storing it
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        timestamp = datetime.now()

        # Insert all the user info into the database
        user_data = {
            "username": username,
            "password": hashed_password,
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "created_at": timestamp
        }

        app.users_collection.insert_one(user_data)
        app.show_popup("✅ Signup successful!")
        app.is_logged_in = True
        app.root.current = "history"
        return True

class HistoryItem(OneLineAvatarIconListItem):
    text = StringProperty()  # Define as a Kivy property
    icon = StringProperty()

    def __init__(self, text="", icon="history", details="", **kwargs):
        super().__init__(text=text, icon=icon, **kwargs)

        #self.add_widget(IconLeftWidget(icon=icon))

class HistoryScreen(Screen):
    def on_pre_enter(self, *args):
        self.load_history()

    def load_history(self):
        app = MDApp.get_running_app()
        history_collection = app.history_collection
        if history_collection is None:
            print("❌ Database connection not available, cannot load history.")
            return

            # Ensure user is logged in
        if not app.current_user_id:
            print("❌ No logged-in user, cannot fetch history.")
            return
        print(f"🔍 Fetching history for user: {app.current_user_id}")  # Debugging
        self.ids.history_list.data = []
            # Fetch history for the logged-in user only
        history_data = list(history_collection.find(
            {"user_id": ObjectId(app.current_user_id)}
        ).sort("timestamp", -1))

        if not history_data:
            print("⚠️ No history found for this user.")
            self.ids.history_list.data = [{"text": "No history available", "icon": "alert"}]
            return

        self.ids.history_list.data = [
            {
                "text": f"{doc['type'].replace('_', ' ').title()} - {datetime.fromisoformat(doc['timestamp']).strftime('%Y-%m-%d %H:%M')}",
                "icon": self.get_icon(doc['type']),
                "on_release": lambda x=doc["_id"]: self.show_history_details(str(x))
            } for doc in history_data
        ]
        print(f"🔄 Loaded {len(history_data)} history items for user {app.current_user_id}.")

    def get_icon(self, activity_type):
        icons = {
            "translate": "translate",
            "object_recognition": "magnify",
            "Color_Detection": "palette",
            "ocr": "file-document-outline",
            "qr_code": "qrcode-scan"
        }
        return icons.get(activity_type, "history")

    def show_filter_options(self, button):
        """Creates and opens a dropdown menu when filter icon is clicked."""
        menu_items = [
            {"text": "All", "on_release": lambda: self.filter_history("All")},
            {"text": "Color Extraction", "on_release": lambda: self.filter_history("Color_Detection")},
            {"text": "QR Code Scan", "on_release": lambda: self.filter_history("qr_code")},
            {"text": "OCR", "on_release": lambda: self.filter_history("ocr")},
            {"text": "Object Recognition", "on_release": lambda: self.filter_history("object_recognition")},
            {"text": "Translation", "on_release": lambda: self.filter_history("translate")},
        ]

        self.menu = MDDropdownMenu(
            caller=button,  # ✅ Attach to the filter icon
            items=menu_items,
            width_mult=2,  # ✅ Adjust width if needed
            position="auto",  # ✅ Ensures better positioning

        )
        self.menu.open()

    def filter_history(self, feature):
        app = MDApp.get_running_app()
        """Retrieves and updates the history list based on the selected feature."""
        if hasattr(self, 'menu'):
            self.menu.dismiss()  # ✅ Close dropdown safely
        rv = self.ids.get("history_list")  # ✅ Correct RecycleView reference

        if not rv:
            print("⚠️ Error: 'history_list' not found in IDs.")
            return

        # ✅ Fetch history data
        collection = app.history_collection

        query = {} if feature.lower() == "all" else {"type": feature}

        history_entries = list(collection.find(query).sort("timestamp", -1))

        print(f"Filtering history with feature: {feature}")
        print(f"Query: {query}")
        print(f"Entries found: {len(history_entries)}")
        if not history_entries:
            rv.data = [{"text": "No history found.", "secondary_text": ""}]  # ✅ Empty case
            return


        # ✅ Format data for RecycleView
        rv.data = [
            {
                "text": f"{doc.get('type', 'Unknown').replace('_', ' ').title()} - {self.format_datetime(doc.get('timestamp', 'N/A'))}",
                "on_release": partial(self.show_history_details, str(doc["_id"]))
            }
            for doc in history_entries
        ]

    def format_datetime(self, date_value):
        """Converts a datetime object or string to 'YYYY-MM-DD h:mm AM/PM' format."""
        if not date_value or date_value == "N/A":
            return "N/A"

        try:
            if isinstance(date_value, datetime):
                dt = date_value  # Already a datetime object
            else:
                dt = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S.%f")  # Convert string to datetime

            return dt.strftime("%Y-%m-%d %I:%M %p")  # Example: 2025-03-07 9:45 PM
        except (ValueError, TypeError):
            return str(date_value)

    def show_history_details(self, history_id):
        """Fetch feature-specific details from MongoDB and update UI card."""
        app = MDApp.get_running_app()
        history_screen = app.root.get_screen("history")
        history_box = history_screen.ids.history_details_box
        history_box.clear_widgets()  # Clear previous details

        history_doc = app.history_collection.find_one({"_id": ObjectId(history_id)})

        if not history_doc:
            print("❌ History item not found.")
            return

        feature = history_doc.get("type")
        reference_id = history_doc.get("reference_id")

        if not reference_id:
            print("❌ No linked feature found in history.")
            return

        # Fetch details based on feature type
        if feature == "translate":
            feature_doc = app.translations_collection.find_one({"_id": ObjectId(reference_id)})
            formatted_date = self.format_datetime(feature_doc.get("created_at", "N/A"))
            details = f"[b]Original:[/b] {feature_doc.get('original_text', 'N/A')}\n" \
                      f"[b]Translated:[/b] {feature_doc.get('translated_text', 'N/A')}\n" \
                      f"[b]Language:[/b] {feature_doc.get('source_language', 'Unknown')} > {feature_doc.get('target_language', 'Unknown')}\n" \
                      f"[b]Date:[/b] {formatted_date}"

        elif feature == "ocr":
            feature_doc = app.ocr_results_collection.find_one({"_id": ObjectId(reference_id)})
            formatted_date = self.format_datetime(feature_doc.get("created_at", "N/A"))
            ocr_text = feature_doc.get("scanned_text", "N/A")
            lines = ocr_text.split("\n")[:3]  # First 3 lines
            preview = "\n".join(lines) + "...\n[b]Tap to expand![/b]" if len(lines) > 3 else "\n".join(lines)
            details = f"[b]Extracted Text:[/b] {preview}\n"\
                      f"[b]Date:[/b] {formatted_date}"

        elif feature == "object_recognition":
            feature_doc = app.objrec_collection.find_one({"_id": ObjectId(reference_id)})
            formatted_date = self.format_datetime(feature_doc.get("created_at", "N/A"))
            details = f"[b]Detected Objects:[/b] {feature_doc.get('detected_object', [])}\n" \
                      f"[b]Date:[/b] {formatted_date}"

        elif feature == "qr_code":
            feature_doc = app.qrcode_collection.find_one({"_id": ObjectId(reference_id)})
            formatted_date = self.format_datetime(feature_doc.get("scan_time", "N/A"))
            details = f"[b]Scanned Data:[/b] {feature_doc.get('scanned_data', 'N/A')}\n" \
                      f"[b]Scan Type:[/b] {feature_doc.get('scan_type', 'Unknown')}\n" \
                      f"[b]Date:[/b] {formatted_date}"

        elif feature == "Color_Detection":
            feature_doc = app.color_extraction_collection.find_one({"_id": ObjectId(reference_id)})
            formatted_date = self.format_datetime(feature_doc.get("extraction_time", "N/A"))
            color_list = feature_doc.get("extracted_colors", [])

            color_details = "\n".join([
                f"[b]Color:[/b] [color={color.get('hex_color', '#FFFFFF')}]{color.get('hex_color', 'N/A')}[/color] ({color.get('percentage', 'N/A')})"
                for color in color_list[:3]  # Show top 3 colors
            ])


            details = f"{color_details}\n[b]Date:[/b] {formatted_date}"

        else:
            print("❌ Unknown feature type.")
            return

        # Card Widget
        card = MDCard(
            height=dp(200),
            padding=dp(10),
            elevation=2,
            md_bg_color=(1, 1, 1, 0.9),
            radius=[dp(10)] * 4,
            size_hint=(1, None)
        )

        # Layout to hold label & close button
        card_layout = BoxLayout(orientation="vertical", spacing=dp(10))



        top_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            padding=[dp(10), dp(5), dp(10), dp(5)]
        )

        # Spacer to push the close button to the right
        top_layout.add_widget(Widget())

        # Close (X) Button
        close_btn = MDIconButton(
            icon="close",
            theme_text_color="Error",
            pos_hint={"right": 1},
            halign="right",
            size_hint=(None, None),
            on_release=lambda x: history_box.remove_widget(card)
        )

        # Label to show history details
        label = MDLabel(
            text=details,
            markup=True,
            halign="left",
            size_hint_y=None,
            height=dp(150),
            padding = [dp(10), 0, dp(10), 0]
        )

        top_layout.add_widget(close_btn)  # Add close button to the top layout

        # Add label and close button to the layout
        card_layout.add_widget(label)
        card_layout.add_widget(top_layout)

        # Add layout to the card
        card.add_widget(card_layout)

        # Add the card to the history box
        history_box.add_widget(card)

        # Navigate to the history details screen
        app.root.current = "history"

class MyApp(MDApp):
    is_logged_in = False
    logged_in_user = None  # Store full user data
    current_user_id = None  # Store user ID as string


    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        db = client["tinyai_db"]
        db.command("ping")  # Test connection
        print("✅ Connected to MongoDB")
        users_collection = db["users"]
        translations_collection = db["translations"]
        ocr_results_collection = db["ocr_results"]
        history_collection = db["activity_history"]
        objrec_collection = db["object_recognition"]
        qrcode_collection = db["qrcode_detection"]
        color_extraction_collection = db["color_extraction"]

    except ServerSelectionTimeoutError:
        print("❌ Failed to connect to MongoDB. Ensure MongoDB is running.")
        client = None  # Prevent further errorsusers_collection = None
        translations_collection = None
        ocr_results_collection = None
        history_collection = None
        objrec_collection = None
        qrcode_collection = None
        color_extraction_collection = None

    def add_to_history(self, user_id, activity_type, description, feature, reference_id=None):
        """Adds a new activity to history in MongoDB."""
        if self.history_collection is None:
            print("❌ Database connection not available, history not saved.")
            return

        history_data = {
            "user_id": ObjectId(user_id),
            "activity": activity_type,
            "type": feature,  # Store feature type
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "reference_id": ObjectId(reference_id) if reference_id else None  # Link to feature data
        }

        try:
            result = self.history_collection.insert_one(history_data)
            print(f"✅ History entry saved with ID: {result.inserted_id}")
            if result.inserted_id:
                print(f"✅ History saved: {history_data}")

        except Exception as e:
            print(f"❌ Error saving history: {e}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logged_in_user = None

    def on_start(self):
        LoginScreen.load_session(self)
        if hasattr(self, "current_user_id") and self.current_user_id:

            username, email = ProfileScreen.get_user_data(user_id=self.current_user_id)
            print(f"🔄 Session restored! Logged-in user: {self.current_user_id}")

            # Get the ProfileScreen directly from the ScreenManager
            profile_screen = self.root.get_screen("profile")  # Replace "profile" with your ProfileScreen name
            Clock.schedule_once(lambda dt:profile_screen.update_user_info(username, email), 0.1)
            # Update the user info on the ProfileScreen
            #profile_screen.update_user_info(username, email)
        else:
            print("⚠️ No active session. Redirecting to login screen.")
            self.root.current = "login"

    def show_popup(self, message):
        popup = Popup(title="Login Failed",
                      content=Label(text=message),
                      size_hint=(0.6, 0.4))
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 3)


    def build(self):
        kv = Builder.load_file("function.kv")
        return kv

    def show_message(self, message):
        print(message)

    def go_back(self, screen_name):
        """Handle back navigation and clear all data from the previous screen."""
        previous_screen = self.root.get_screen(screen_name)  # Get the previous screen dynamically

        dynamic_attributes = ["image_path", "detected_text", "detected_objects", "result_text"]

        for attr in dynamic_attributes:
            if hasattr(previous_screen, attr):
                setattr(previous_screen, attr, "" if isinstance(getattr(previous_screen, attr), str) else None)

        # Loop through all widgets inside the screen
        for widget in previous_screen.walk():
            # Clear text input fields
            if isinstance(widget, (MDTextField, TextInput)):
                widget.text = ""

            # Reset dropdown (Spinner)
            elif isinstance(widget, Spinner):
                widget.text = widget.values[0] if widget.values else ""

            # Clear MDCard contents but keep the card itself
            elif isinstance(widget, MDCard):
                if widget.parent:
                    widget.parent.remove_widget(widget)


            elif isinstance(widget, Image) and hasattr(widget, 'source'):
                widget.source = ""

            elif screen_name == "ColorExtraction" and isinstance(widget,BoxLayout) and widget == previous_screen.ids.color_palette:
                widget.clear_widgets()  # Clear only color buttons inside the BoxLayout


        # Navigate back to the main screen
        self.root.current = 'main'

    def go_to_screen(self, screen_name):
        self.root.current = screen_name

    def switch_to_login(self):
        self.root.current = "login"

    def open_image_url(self, img_url):
        """ Opens the website where the image is hosted. """
        webbrowser.open(img_url)  # Opens the original image link

    def copy_text(self, extracted_text):
        """ Copies the extracted OCR text to the clipboard. """
        if extracted_text:
            pyperclip.copy(extracted_text)  # Copy the text
            toast("Text Copied")
        else:
            toast("No text found to copy.")

    def open_history(self):
        """Function to handle history navigation."""
        print("Opening History...")  # Replace with actual navigation logic

    def change_theme(self):
        """Function to handle theme change."""
        print("Changing Theme...")  # Implement dark/light mode toggle

    def show_about(self):
        """Function to show the About section."""
        print("Showing About...")  # Replace with dialog or new screen

    def logout(self):
        """Clear user session and log out."""
        app = MDApp.get_running_app()
        app.is_logged_in = False
        app.logged_in_user = None

        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

        print("✅ Logged out successfully")  # Implement logout logic, such as clearing session data




if __name__ == "__main__":
    MyApp().run()

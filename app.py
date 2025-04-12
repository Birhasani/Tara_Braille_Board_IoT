import streamlit as st
import subprocess
import logging
import os
from PIL import Image
import easyocr
import numpy as np
from google import genai

class TaraDashboard:
    def __init__(self):
        self.api_key = st.secrets["general"]["api_key"]
        self.google_credentials = st.secrets["general"]["google_application_credentials"]
        self.model_path = r"pretrained_model_tts\best_model.pth"  # Path ke model TTS
        self.config_path = r"pretrained_model_tts\config.json"   # Path ke config TTS
        self.output_path = "output.wav"  # Output file untuk suara
        self.speakers_path = r"pretrained_model_tts\speakers.pth"  # Path speakers
        self.client = genai.Client(api_key=self.api_key)  # Inisialisasi Gemini API Client

        # Inisialisasi EasyOCR Reader untuk Bahasa Indonesia dan Bahasa Inggris
        self.reader = easyocr.Reader(['en', 'id'])  # Menambahkan lebih banyak bahasa jika perlu

    def detect_text_from_image(self, image):
        """Deteksi teks dari gambar menggunakan EasyOCR"""
        # Mengonversi PIL Image ke numpy array
        image_np = np.array(image)

        # Menggunakan EasyOCR untuk mendeteksi teks
        result = self.reader.readtext(image_np)
        
        # Menggabungkan teks yang terdeteksi menjadi satu string
        detected_text = " ".join([text[1] for text in result])
        
        return detected_text if detected_text else ""

    def summarize_text(self, input_text):
        """Ringkas teks menggunakan Gemini API"""
        chat = self.client.chats.create(model="gemini-2.0-flash")
        response = chat.send_message(f"Ringkas secara terstruktur dari teks berikut menjadi pengetahuan yang mudah untuk dipahami dengan menggunakan bahasa indonesia, akan tetapi jika dideteksi tidak perlu diringkas jangan diringkas pengetahuan lengkapnya, nanti outputnya kasih awalan berikut merupakan ringkasan di papan tulis : {input_text}")
        return response.text

    def convert_text_to_speech(self, summarized_text, speaker_name="wibowo"):
        """Konversi teks menjadi suara menggunakan TTS"""
        tts_command = [
            "tts", "--text", summarized_text,
            "--model_path", self.model_path,
            "--config_path", self.config_path,
            "--speaker_idx", speaker_name,  # Menggunakan nama speaker yang dipilih
            "--out_path", self.output_path
        ]
        try:
            result = subprocess.run(tts_command, check=True, text=True, capture_output=True)
            st.write("Standard Error:")
            st.write(result.stderr)
            st.success("Speech generated successfully!")
            st.audio(self.output_path, format="audio/wav")
        except subprocess.CalledProcessError as e:
            st.error(f"Error generating speech: {e}")
            st.write("Error Output:")
            st.write(e.stderr)

    def run(self):
        """Menjalankan aplikasi Streamlit"""
        st.title("Tara Dashboard")
        
        # Input gambar untuk deteksi teks
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        # Pilihan speaker
        available_speakers = ["wibowo", "ardi", "gadis"]  # Daftar speaker
        speaker_name = st.selectbox("Choose Speaker", available_speakers)  # Pilih speaker

        if uploaded_image:
            # Menampilkan gambar yang di-upload
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Processing Image..."):
                detected_text = self.detect_text_from_image(image)
            
            if detected_text:
                st.write("Detected Text from Image:")
                st.write(detected_text)

                # Menambahkan tombol untuk merangkum teks
                if st.button("Summarize Text"):
                    summarized_text = self.summarize_text(detected_text)
                    st.write("Summarized Text:")
                    st.write(summarized_text)  # Menampilkan teks ringkasan
                    
                    # Konversi teks yang diringkas menjadi suara dengan speaker yang dipilih
                    self.convert_text_to_speech(summarized_text, speaker_name)
            else:
                st.warning("No text detected in the image.")
        else:
            st.warning("Please upload an image to detect text.")

if __name__ == "__main__":
    dashboard = TaraDashboard()  # Membuat objek TaraDashboard
    dashboard.run()  # Menjalankan aplikasi

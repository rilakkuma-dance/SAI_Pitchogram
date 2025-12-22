#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import speech_recognition as sr
import whisper
import threading
from datetime import datetime
import queue
import numpy as np


class SimpleTranscriptionApp:
    def __init__(self):
        # Initialize components
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcription_text_content = ""
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 1000
        self.recognizer.dynamic_energy_threshold = False
        self.microphone = sr.Microphone(sample_rate=16000)
        
        # Whisper model (will be loaded when needed)
        self.audio_model = None
        self.current_model = None
        
        # Setup GUI
        self.setup_gui()
        
        # Adjust for ambient noise
        with self.microphone:
            self.recognizer.adjust_for_ambient_noise(self.microphone)
    
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Chinese Speech Transcription")
        self.root.geometry("700x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="medium")  # Default to medium for better Chinese recognition
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"],
                                  state="readonly", width=10)
        model_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        # Language selection with Chinese as default
        ttk.Label(control_frame).pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="zh")  # Default to Chinese
        
        # Record button
        self.record_button = ttk.Button(control_frame, text="🎤 Start Recording", 
                                       command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready for Chinese speech", foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        # Transcription area
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(text_frame, text="Chinese Transcription:").pack(anchor=tk.W)
        
        # Use a font that supports Chinese characters
        self.text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                  font=("SimHei", 12), height=20)  # Chinese font
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_text).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Save", command=self.save_text).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="Copy", command=self.copy_text).pack(side=tk.LEFT, padx=(10, 0))
    
    def load_model(self, model_name):
        """Load Whisper model optimized for Chinese"""
        if self.current_model != model_name:
            self.status_label.config(text="Loading model for Chinese...", foreground="orange")
            self.root.update()
            
            try:
                # Load the model
                self.audio_model = whisper.load_model(model_name)
                self.current_model = model_name
                
                # Give recommendation for Chinese recognition
                if model_name in ["tiny", "base"]:
                    print(f"Note: For better Chinese recognition, consider using 'medium' or 'large' model")
                
                self.status_label.config(text=f"Model {model_name} loaded for Chinese", foreground="green")
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.status_label.config(text="Model load failed", foreground="red")
                return False
        return True
    
    def toggle_recording(self):
        """Start/stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording Chinese speech"""
        model_name = self.model_var.get()
        if not self.load_model(model_name):
            return
        
        self.is_recording = True
        self.record_button.config(text="🛑 Stop Recording")
        self.status_label.config(text="Recording Chinese speech...", foreground="red")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording")
        self.status_label.config(text="Stopped", foreground="orange")
    
    def record_audio(self):
        """Record audio in chunks"""
        try:
            with self.microphone as source:
                while self.is_recording:
                    try:
                        # Listen for audio with timeout - slightly longer for Chinese speech
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                        self.audio_queue.put(audio)
                    except sr.WaitTimeoutError:
                        continue
        except Exception as e:
            print(f"Recording error: {e}")
    
    def process_audio(self):
        """Process recorded audio with Chinese language model"""
        while self.is_recording:
            try:
                # Get audio from queue
                try:
                    audio = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Convert audio to numpy array
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                
                # Always use Chinese language model
                result = self.audio_model.transcribe(
                    audio_data, 
                    language="th",  # Always Chinese
                    task="transcribe"
                )
                
                text = result['text'].strip()
                
                # Update GUI with Chinese text
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.root.after(0, self.add_transcription, f"[{timestamp}] {text}")
                    
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def add_transcription(self, text):
        """Add Chinese text to transcription area"""
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)
        self.transcription_text_content += text + "\n"
    
    def clear_text(self):
        """Clear transcription"""
        self.text_area.delete(1.0, tk.END)
        self.transcription_text_content = ""
    
    def save_text(self):
        """Save Chinese transcription to file"""
        if not self.transcription_text_content.strip():
            messagebox.showwarning("Warning", "No transcription to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Save with UTF-8 encoding to preserve Chinese characters
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.transcription_text_content)
                messagebox.showinfo("Success", f"Chinese transcription saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def copy_text(self):
        """Copy Chinese transcription to clipboard"""
        if not self.transcription_text_content.strip():
            messagebox.showwarning("Warning", "No transcription to copy!")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(self.transcription_text_content)
        messagebox.showinfo("Success", "Chinese transcription copied to clipboard!")
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """Handle window closing"""
        if self.is_recording:
            self.stop_recording()
        self.root.destroy()


def main():
    """Main function"""
    print("Starting Chinese Speech Transcription App...")
    print("Default language: Chinese (zh)")
    print("Recommended model: medium or large for better Chinese recognition")
    app = SimpleTranscriptionApp()
    app.run()


if __name__ == "__main__":
    main()
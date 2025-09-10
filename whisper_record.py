#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import threading


class TranscriptionApp:
    def __init__(self, args):
        self.args = args
        self.is_recording = False
        self.phrase_time = None
        self.data_queue = Queue()
        self.phrase_bytes = bytes()
        self.transcription = ['']
        
        # Initialize speech recognizer
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        
        # Setup microphone
        self.setup_microphone()
        
        # Load Whisper model
        model = args.model
        if args.model != "large" and not args.non_english:
            model = model + ".en"
        print(f"Loading Whisper model: {model}")
        self.audio_model = whisper.load_model(model)
        print("Model loaded successfully!")
        
        # Setup GUI
        self.setup_gui()
        
        # Background thread for processing audio
        self.processing_thread = None
        self.stop_processing = False
        
    def setup_microphone(self):
        """Setup microphone based on platform"""
        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)
            
        # Adjust for ambient noise
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Real-time Speech Transcription")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Record button
        self.record_button = ttk.Button(
            main_frame, 
            text="Start Recording", 
            command=self.toggle_recording,
            style="Record.TButton"
        )
        self.record_button.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to record", foreground="green")
        self.status_label.grid(row=0, column=2, padx=(10, 0), pady=(0, 10))
        
        # Transcription display
        ttk.Label(main_frame, text="Transcription:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.transcription_text = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=25,
            font=("Consolas", 11)
        )
        self.transcription_text.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear button
        clear_button = ttk.Button(main_frame, text="Clear", command=self.clear_transcription)
        clear_button.grid(row=3, column=0, pady=(10, 0), sticky=tk.W)
        
        # Save button
        save_button = ttk.Button(main_frame, text="Save", command=self.save_transcription)
        save_button.grid(row=3, column=1, pady=(10, 0), sticky=tk.W, padx=(10, 0))
        
        # Style configuration
        style = ttk.Style()
        style.configure("Record.TButton", font=("Arial", 12, "bold"))
        
    def record_callback(self, _, audio: sr.AudioData) -> None:
        """Threaded callback function to receive audio data when recordings finish."""
        if self.is_recording:
            data = audio.get_raw_data()
            self.data_queue.put(data)
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording and processing"""
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_label.config(text="Recording...", foreground="red")
        
        # Clear previous data
        self.phrase_time = None
        self.phrase_bytes = bytes()
        self.data_queue = Queue()
        
        # Start background recording
        self.recorder.listen_in_background(
            self.source, 
            self.record_callback, 
            phrase_time_limit=self.args.record_timeout
        )
        
        # Start processing thread
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_recording(self):
        """Stop recording and processing"""
        self.is_recording = False
        self.stop_processing = True
        self.record_button.config(text="Start Recording")
        self.status_label.config(text="Stopped", foreground="orange")
        
        # Stop background recording
        self.recorder.stop_listening_for_all()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def process_audio(self):
        """Process audio data in background thread"""
        while not self.stop_processing:
            try:
                now = datetime.now()
                
                if not self.data_queue.empty():
                    phrase_complete = False
                    
                    # Check if phrase is complete
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
                        self.phrase_bytes = bytes()
                        phrase_complete = True
                    
                    self.phrase_time = now
                    
                    # Combine audio data from queue
                    audio_data = b''.join(list(self.data_queue.queue))
                    self.data_queue.queue.clear()
                    
                    # Add new audio data
                    self.phrase_bytes += audio_data
                    
                    # Convert to format for Whisper
                    audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe
                    result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    
                    # Update transcription
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text
                    
                    # Update GUI in main thread
                    self.root.after(0, self.update_transcription_display)
                else:
                    sleep(0.1)
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
                break
    
    def update_transcription_display(self):
        """Update the transcription display"""
        self.transcription_text.delete(1.0, tk.END)
        for line in self.transcription:
            if line.strip():  # Only show non-empty lines
                self.transcription_text.insert(tk.END, line + "\n")
        self.transcription_text.see(tk.END)  # Scroll to bottom
    
    def clear_transcription(self):
        """Clear the transcription"""
        self.transcription = ['']
        self.transcription_text.delete(1.0, tk.END)
    
    def save_transcription(self):
        """Save transcription to file"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for line in self.transcription:
                        if line.strip():
                            f.write(line + "\n")
                self.status_label.config(text=f"Saved to {filename}", foreground="blue")
                self.root.after(3000, lambda: self.status_label.config(text="Ready", foreground="green"))
            except Exception as e:
                self.status_label.config(text=f"Error saving: {e}", foreground="red")
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_recording:
            self.stop_recording()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Create and run the application
    app = TranscriptionApp(args)
    app.run()


if __name__ == "__main__":
    main()
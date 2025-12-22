import pyaudio
import wave
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import time

class RecordAndTranscribeGUI:
    def __init__(self):
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        
        # Recording variables
        self.is_recording = False
        self.frames = []
        self.audio = None
        self.stream = None
        self.last_saved_file = None
        
        # Setup save directory
        self.save_dir = self.setup_save_directory()
        
        # Create GUI
        self.setup_gui()
        
    def setup_save_directory(self):
        """Create save directory in Documents folder"""
        save_path = Path.home() / "Documents" / "AudioRecordings"
        save_path.mkdir(exist_ok=True)
        return save_path
    
    def setup_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Record & Transcribe Audio")
        self.root.geometry("450x500")
        self.root.resizable(False, False)
        
        # Title
        title_label = tk.Label(self.root, text="Record & Transcribe", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=15)
        
        # Settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(pady=10)
        
        # Language selection
        tk.Label(settings_frame, text="Language:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.language_var = tk.StringVar(value="auto")
        language_combo = ttk.Combobox(settings_frame, textvariable=self.language_var, 
                                     values=["auto", "en", "th", "es", "fr", "de"], width=10)
        language_combo.grid(row=0, column=1, padx=5)
        
        # Model selection
        tk.Label(settings_frame, text="Model:", font=("Arial", 10)).grid(row=0, column=2, sticky="w", padx=(10,0))
        self.model_var = tk.StringVar(value="turbo")
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var,
                                  values=["turbo", "base", "small", "medium", "large"], width=10)
        model_combo.grid(row=0, column=3, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to record", 
                                    font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=15)
        
        # Record button
        self.record_btn = tk.Button(self.root, text="🎙️ Start Recording", 
                                   font=("Arial", 14, "bold"),
                                   bg="#4CAF50", fg="white",
                                   width=20, height=2,
                                   command=self.start_recording)
        self.record_btn.pack(pady=10)
        
        # Stop button
        self.stop_btn = tk.Button(self.root, text="⏹️ Stop & Transcribe", 
                                 font=("Arial", 14, "bold"),
                                 bg="#f44336", fg="white",
                                 width=20, height=2,
                                 command=self.stop_and_transcribe,
                                 state=tk.DISABLED)
        self.stop_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill='x')
        
        # Recording time label
        self.time_label = tk.Label(self.root, text="", 
                                  font=("Arial", 12, "bold"), fg="red")
        self.time_label.pack(pady=5)
        
        # Result text area
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(result_frame, text="Transcription Result:", font=("Arial", 10, "bold")).pack(anchor='w')
        
        self.result_text = tk.Text(result_frame, height=8, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Save location info
        location_label = tk.Label(self.root, 
                                 text=f"Files saved to: {self.save_dir}",
                                 font=("Arial", 8), fg="gray")
        location_label.pack(pady=5)
        
        # Start time tracking
        self.start_time = None
        self.update_timer()
        
    def start_recording(self):
        """Start recording audio"""
        try:
            self.frames = []
            self.is_recording = True
            self.start_time = datetime.now()
            
            # Clear previous results
            self.result_text.delete(1.0, tk.END)
            
            # Setup audio
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Update GUI
            self.status_label.config(text="Recording... Speak now!", fg="red")
            self.record_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not start recording: {e}")
    
    def record_audio(self):
        """Record audio in background thread"""
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK_SIZE)
                self.frames.append(data)
            except:
                break
    
    def stop_and_transcribe(self):
        """Stop recording and immediately transcribe"""
        if not self.is_recording:
            return
            
        # Stop recording
        self.is_recording = False
        
        # Update status
        self.status_label.config(text="Saving recording...", fg="orange")
        self.progress.start()
        
        # Wait for recording thread to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        
        # Update GUI
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.time_label.config(text="")
        
        # Save recording and transcribe
        filename = self.save_recording()
        
        if filename:
            self.transcribe_audio(filename)
        else:
            self.progress.stop()
            self.status_label.config(text="Recording failed!", fg="red")
    
    def save_recording(self):
        """Save recorded audio to file"""
        if not self.frames:
            messagebox.showwarning("Warning", "No audio to save!")
            return None
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        file_path = self.save_dir / filename
        
        try:
            # Save as WAV file
            with wave.open(str(file_path), 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(b''.join(self.frames))
            
            return file_path
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")
            return None
    
    def transcribe_audio(self, audio_file):
        """Transcribe the recorded audio"""
        def transcribe_worker():
            try:
                # Import here to avoid startup delay
                import whisper
                import librosa
                
                self.status_label.config(text="Loading audio...", fg="orange")
                
                # Load audio
                audio, sr = librosa.load(str(audio_file), sr=16000)
                duration = len(audio) / sr
                
                self.status_label.config(text=f"Loading {self.model_var.get()} model...", fg="orange")
                
                # Load model
                model = whisper.load_model(self.model_var.get())
                
                self.status_label.config(text="Transcribing... Please wait", fg="orange")
                
                # Transcribe
                language = self.language_var.get()
                if language == "auto":
                    result = model.transcribe(audio, fp16=False)
                else:
                    result = model.transcribe(audio, language=language, fp16=False)
                
                # Update GUI in main thread
                self.root.after(0, self.display_results, result, audio_file, duration)
                
            except ImportError:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    "Whisper not installed!\nRun: pip install openai-whisper librosa"))
                self.root.after(0, self.reset_gui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Transcription Error", str(e)))
                self.root.after(0, self.reset_gui)
        
        # Start transcription in background thread
        transcribe_thread = threading.Thread(target=transcribe_worker)
        transcribe_thread.start()
    
    def display_results(self, result, audio_file, duration):
        """Display transcription results"""
        self.progress.stop()
        
        # Display transcription
        transcription = result['text'].strip()
        detected_language = result.get('language', 'Unknown')
        
        # Update result text
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Language: {detected_language}\n")
        self.result_text.insert(tk.END, f"Duration: {duration:.1f}s\n")
        self.result_text.insert(tk.END, "-" * 40 + "\n")
        self.result_text.insert(tk.END, transcription)
        
        # Save transcription to file
        self.save_transcription(result, audio_file)
        
        # Update status
        self.status_label.config(text="✅ Recording & Transcription Complete!", fg="green")
        
        # Show completion message
        messagebox.showinfo("Complete!", 
                           f"Audio recorded and transcribed successfully!\n\n"
                           f"Language: {detected_language}\n"
                           f"Duration: {duration:.1f} seconds\n\n"
                           f"Files saved to:\n{self.save_dir}")
    
    def save_transcription(self, result, audio_file):
        """Save transcription to text file"""
        try:
            audio_path = Path(audio_file)
            output_file = audio_path.parent / f"{audio_path.stem}_transcription.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("AUDIO TRANSCRIPTION\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Audio file: {audio_file.name}\n")
                f.write(f"Language: {result.get('language', 'Unknown')}\n")
                f.write(f"Model: {self.model_var.get()}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("TRANSCRIPTION:\n")
                f.write("-" * 20 + "\n")
                f.write(result['text'] + "\n")
                
        except Exception as e:
            print(f"Error saving transcription: {e}")
    
    def reset_gui(self):
        """Reset GUI to initial state"""
        self.progress.stop()
        self.status_label.config(text="Ready to record", fg="blue")
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def update_timer(self):
        """Update recording timer"""
        if self.is_recording and self.start_time:
            elapsed = datetime.now() - self.start_time
            seconds = int(elapsed.total_seconds())
            mins, secs = divmod(seconds, 60)
            self.time_label.config(text=f"Recording: {mins:02d}:{secs:02d}")
        
        # Update every second
        self.root.after(1000, self.update_timer)
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            if self.is_recording:
                self.is_recording = False

def main():
    """Main function"""
    print("Record & Transcribe Audio")
    print("=" * 40)
    print("Requirements: pip install pyaudio openai-whisper librosa")
    print("Starting application...")
    
    # Check basic requirements
    try:
        import pyaudio
        import tkinter
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install pyaudio")
        return
    
    # Create and run the application
    app = RecordAndTranscribeGUI()
    app.run()

if __name__ == "__main__":
    main()
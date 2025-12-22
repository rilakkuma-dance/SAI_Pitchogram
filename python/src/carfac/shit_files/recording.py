import pyaudio
import wave
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

class AudioRecorderGUI:
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
        self.root.title("Audio Recorder")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Title
        title_label = tk.Label(self.root, text="Audio Recorder", 
                              font=("Arial", 18, "bold"))
        title_label.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to record", 
                                    font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=10)
        
        # Record button
        self.record_btn = tk.Button(self.root, text="🎙️ Record", 
                                   font=("Arial", 14, "bold"),
                                   bg="#4CAF50", fg="white",
                                   width=15, height=2,
                                   command=self.start_recording)
        self.record_btn.pack(pady=10)
        
        # Stop button
        self.stop_btn = tk.Button(self.root, text="⏹️ Stop", 
                                 font=("Arial", 14, "bold"),
                                 bg="#f44336", fg="white",
                                 width=15, height=2,
                                 command=self.stop_recording,
                                 state=tk.DISABLED)
        self.stop_btn.pack(pady=10)
        
        # Save location info
        location_label = tk.Label(self.root, 
                                 text=f"Files saved to:\n{self.save_dir}",
                                 font=("Arial", 9), fg="gray")
        location_label.pack(pady=20)
        
        # Recording time label
        self.time_label = tk.Label(self.root, text="", 
                                  font=("Arial", 12, "bold"), fg="red")
        self.time_label.pack()
        
        # Start time tracking
        self.start_time = None
        self.update_timer()
        
    def start_recording(self):
        """Start recording audio"""
        try:
            self.frames = []
            self.is_recording = True
            self.start_time = datetime.now()
            
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
            self.status_label.config(text="Recording...", fg="red")
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
    
    def stop_recording(self):
        """Stop recording and save file"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        
        # Save recording
        filename = self.save_recording()
        
        # Update GUI
        self.status_label.config(text="Recording saved!", fg="green")
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.time_label.config(text="")
        
        if filename:
            messagebox.showinfo("Success", f"Recording saved as:\n{filename}")
    
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
                self.stop_recording()

def main():
    """Main function"""
    # Check if required packages are installed
    try:
        import pyaudio
        import tkinter
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install pyaudio")
        return
    
    # Create and run the recorder
    recorder = AudioRecorderGUI()
    recorder.run()

if __name__ == "__main__":
    main()
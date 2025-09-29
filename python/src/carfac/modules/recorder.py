import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import queue

import threading
import traceback

class AudioRecorder:
    """Recording audio with state management."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.enabled = True
        
        # Audio buffer for mic recording
        self.audio_buffer = []

        # State variables
        self.is_recording = False
        self.is_processing = False
        self.result = None

        # Callback system
        self.audio_callbacks = []
        self.processing_queue = queue.Queue()

        try:
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            self.recognizer = sr.Recognizer()
            with self.microphone:
                self.recognizer.adjust_for_ambient_noise(self.microphone)
            print("Microphone ready")
        except Exception as e:
            print(f"Microphone initialization warning: {e}")
            self.microphone = None
            self.recognizer = None

    def add_audio_callback(self, callback, *callback_args):
        """Add a callback to be called with new audio data"""
        self.audio_callbacks.append((callback, callback_args))

    def remove_audio_callback(self, callback):
        """Remove a previously added callback"""
        self.audio_callbacks = [cb for cb in self.audio_callbacks if cb[0] != callback]

    def get_current_status(self):
        """Simple 3-state status"""
        if self.is_recording:
            return "Recording...", "yellow"
        
        if self.is_processing:
            return "Processing phonemes...", "orange"
        
        return "Ready to start", "blue"

    def start_recording(self):
        """Start recording"""
        if not self.enabled or self.is_recording:
            return False
        
        # Reset state
        self.is_recording = True
        self.is_processing = False
        self.result = None
        self.audio_data = []
        
        # Start recording thread
        threading.Thread(target=self._record_audio, daemon=True).start()
        return True

    def stop_recording(self):
        """Stop recording and process"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.is_processing = True
        
        # Start processing thread
        threading.Thread(target=self._process_audio, daemon=True).start()

    def get_current_result(self):
        """Get the latest transcription result"""
        return self.result if self.result else "no_audio"

    def _record_audio(self):
        """Record audio continuously"""
        try:
            with self.microphone as source:
                stream = source.stream
                while self.is_recording:
                    # Read a small chunk of audio continuously
                    audio_chunk = stream.read(source.CHUNK)
                    audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_data.append(audio_np)

        except Exception as e:
            print(f"Recording error: {e}")

    def _run_callbacks(self, complete_audio):
        """Run all registered audio callbacks"""
        for callback, args in self.audio_callbacks:
            try:
                threading.Thread(target=callback, args=(complete_audio, *args), daemon=True).start()
            except Exception as e:
                print(f"Callback error: {e}")
                traceback.print_exc()

    def _process_audio(self):
        """Process all recorded audio with Wav2Vec2"""
        try:
            if not self.audio_data:
                self.result = "no_audio"
                self.is_processing = False
                return
            
            # Combine all audio
            complete_audio = np.concatenate(self.audio_data)
            self._run_callbacks(complete_audio)
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            traceback.print_exc()
            self.result = "no_audio"
            self.analysis_results = None
            self.overall_score = 0.0
        
        self.is_processing = False

    def _save_audio(self, audio_data, filename):
        """Save audio to file"""
        try:
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Convert to 16-bit
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"Audio saved: {filename}")
        except Exception as e:
            print(f"Save error: {e}")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
import subprocess
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import threading
import random
import os
from datetime import datetime
import time

class ToneIntroductionQuiz:
    def __init__(self, audio_base_path=None):
        # Auto-detect audio path relative to script location
        if audio_base_path is None:
            script_dir = Path(__file__).parent.resolve()
            
            # Try multiple possible locations
            possible_paths = [
                script_dir / 'reference',  # Same directory as script
                script_dir.parent / 'reference',  # One level up
                script_dir / 'carfac' / 'reference',  # In carfac subdirectory
            ]
            
            # Find the first path that exists
            audio_base_path = None
            for path in possible_paths:
                if path.exists():
                    audio_base_path = path
                    print(f"✓ Found audio path: {audio_base_path}")
                    break
            
            if audio_base_path is None:
                print(f"⚠️ Warning: Could not find reference audio folder!")
                print(f"   Tried these locations:")
                for path in possible_paths:
                    print(f"   - {path}")
                audio_base_path = script_dir / 'reference'  # Default fallback
        
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = 16000
        self.is_playing = False
        
        # All words from VocabList (IDs 1-15)
        self.vocab_items = [
            {"id": 1, "chinese": "书", "pinyin": "shū", "tone": "1", "audio": "men/1_men.wav"},
            {"id": 2, "chinese": "女人", "pinyin": "nǚrén", "tone": "32", "audio": "women/2_women.wav"},
            {"id": 3, "chinese": "雄", "pinyin": "xióng", "tone": "2", "audio": "men/3_men.wav"},
            {"id": 4, "chinese": "去", "pinyin": "qù", "tone": "4", "audio": "men/4_men.wav"},
            {"id": 6, "chinese": "喜欢", "pinyin": "xǐhuān", "tone": "31", "audio": "women/6_women.wav"},
            {"id": 7, "chinese": "街道", "pinyin": "jiēdào", "tone": "14", "audio": "women/7_women.wav"},
            {"id": 8, "chinese": "熊猫", "pinyin": "xióngmāo", "tone": "21", "audio": "men/8_men.wav"},
            {"id": 9, "chinese": "书店", "pinyin": "shūdiàn", "tone": "14", "audio": "women/9_women.wav"},
            {"id": 10, "chinese": "去年", "pinyin": "qùnián", "tone": "42", "audio": "men/10_men.wav"},
            {"id": 11, "chinese": "中午", "pinyin": "zhōngwǔ", "tone": "13", "audio": "women/11_women.wav"},
            {"id": 12, "chinese": "椅子", "pinyin": "yǐzi", "tone": "35", "audio": "men/12_men.wav"},
            {"id": 13, "chinese": "学校", "pinyin": "xuéxiào", "tone": "24", "audio": "women/13_women.wav"},
            {"id": 14, "chinese": "医院", "pinyin": "yīyuàn", "tone": "14", "audio": "men/14_men.wav"},
            {"id": 15, "chinese": "游戏", "pinyin": "yóuxì", "tone": "24", "audio": "women/15_women.wav"},
            {"id": 16, "chinese": "她", "pinyin": "tā", "tone": "1", "audio": "men/16_men.wav"},
        ]
        
        # Verify audio path exists
        if not self.audio_base_path.exists():
            print(f"⚠️ Warning: Audio path does not exist: {self.audio_base_path}")
            print(f"   Please ensure 'reference' folder exists with audio files")
        else:
            print(f"✓ Using audio path: {self.audio_base_path}")
        
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        
        # Timer variables (hidden from display, only for recording)
        self.question_start_time = None
        self.question_elapsed_time = 0
        self.timer_started = False
        
        # Store results for each question
        self.results = []
        self.session_start_time = datetime.now()
        
        self.fig = plt.figure(figsize=(6, 8))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
        self._select_random_item()
        
    def _setup_interface(self):
        # Main container
        main_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        main_ax.set_xlim(0, 1)
        main_ax.set_ylim(0, 1)
        main_ax.axis('off')
        
        # Title
        main_ax.text(0.5, 0.95, 'Mandarin has 4 tones as below:', 
                    fontsize=18, ha='center', va='top', weight='bold')
        
        # Tone visualization area
        viz_ax = self.fig.add_axes([0.2, 0.60, 0.6, 0.25])
        viz_ax.set_xlim(0, 1)
        viz_ax.set_ylim(0, 4)
        viz_ax.axis('off')
        
        # Progress counter (top right) - moved to where timer was
        self.progress_text = main_ax.text(0.5, 1.00, '', 
                    fontsize=12, ha='center', va='top', weight='bold',
                    color='#7f8c8d')
        
        # Play button
        ax_play = plt.axes([0.35, 0.38, 0.3, 0.06])
        self.btn_play = Button(ax_play, '▶ Play', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.play_audio)
        
        # Status text (for audio status only)
        self.status_text = main_ax.text(0.5, 0.32, 'Click Play to hear the word', 
                    fontsize=10, ha='center', va='center', color='#7f8c8d')
        
        # Instruction text
        main_ax.text(0.5, 0.26, 'Type the correct tones number', 
                    fontsize=11, ha='center', va='center', color='#666666')
        
        main_ax.text(0.5, 0.20, 'Example: tiānqì -> 14', 
                    fontsize=11, ha='center', va='center', color='#666666')
        
        # Text input box
        ax_input = plt.axes([0.2, 0.18, 0.6, 0.06])
        self.text_input = TextBox(ax_input, '', initial='', 
                                 color='white', hovercolor='#f9f9f9')
        
        # Answer display text (shows what user typed)
        self.answer_text = main_ax.text(0.5, 0.12, '', 
                    fontsize=14, ha='center', va='center', weight='bold',
                    color='#34495e')
        
        # Feedback text (shows correct/incorrect) - positioned below answer
        self.feedback_text = main_ax.text(0.5, 0.07, '', 
                    fontsize=18, ha='center', va='center', weight='bold')
        
        # Check Answer button
        ax_check = plt.axes([0.15, 0.01, 0.3, 0.05])
        self.btn_check = Button(ax_check, 'Check Answer', color='#3498db', hovercolor='#2980b9')
        self.btn_check.label.set_color('white')
        self.btn_check.on_clicked(self.check_answer_button)
        
        # Next Word button
        ax_next = plt.axes([0.55, 0.01, 0.3, 0.05])
        self.btn_next = Button(ax_next, 'Next Word', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)
        
        # Update progress counter
        self._update_progress()
        
    def _update_progress(self):
        """Update the progress counter"""
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        self.fig.canvas.draw_idle()
        
    def _select_random_item(self):
        """Select a random vocabulary item"""
        self.current_item = random.choice(self.vocab_items)
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        
        self.status_text.set_text('Click Play to hear the word')
        self.status_text.set_color('#7f8c8d')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.text_input.set_val('')
        
        self.fig.canvas.draw_idle()
        
        # Update progress display
        self._update_progress()
        
        # Debug: print selected item info
        print(f"\n{'='*60}")
        print(f"NEW WORD SELECTED (Question {self.question_count + 1}/{self.max_questions})")
        print(f"{'='*60}")
        print(f"Chinese: {self.current_item['chinese']}")
        print(f"Pinyin: {self.current_item['pinyin']}")
        print(f"Correct tone: {self.current_item['tone']}")
        print(f"ID: {self.current_item['id']}")
        print(f"Timer will start when Play button is clicked (hidden from display)")
        print(f"{'='*60}")
        
    def _play_audio_file(self, audio_path):
        """Play a single audio file"""
        try:
            if not audio_path.exists():
                print(f"⚠️ Audio file not found: {audio_path}")
                self.status_text.set_text(f"⚠️ Audio file not found")
                self.status_text.set_color('red')
                self.fig.canvas.draw_idle()
                return False
            
            # Print what's being played
            print(f"\n🔊 PLAYING AUDIO:")
            print(f"   File: {audio_path.name}")
            print(f"   Full path: {audio_path}")
            print(f"   Chinese: {self.current_item['chinese']}")
            print(f"   Pinyin: {self.current_item['pinyin']}")
            print(f"   Tone: {self.current_item['tone']}")
            
            audio_data, sr = sf.read(str(audio_path))
            duration = len(audio_data) / sr
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Sample rate: {sr} Hz")
            
            sd.play(audio_data, sr)
            sd.wait()
            
            print(f"✓ Playback complete")
            print(f"⏱ Timer started (hidden)\n")
            return True
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            self.status_text.set_text(f"❌ Error: {str(e)[:30]}")
            self.status_text.set_color('red')
            self.fig.canvas.draw_idle()
            return False
    
    def play_audio(self, event):
        """Play the current word's audio and start timer"""
        if self.is_playing or not self.current_item:
            return
        
        def _play():
            self.is_playing = True
            self.btn_play.label.set_text('Playing...')
            self.status_text.set_text('🔊 Playing audio...')
            self.status_text.set_color('#3498db')
            self.fig.canvas.draw_idle()
            
            audio_path = self.audio_base_path / self.current_item['audio']
            success = self._play_audio_file(audio_path)
            
            if success:
                # Start timer AFTER audio finishes playing (hidden from display)
                self.question_start_time = time.time()
                self.timer_started = True
                
                self.status_text.set_text('Ready for your answer')
                self.status_text.set_color('#27ae60')
            
            self.is_playing = False
            self.btn_play.label.set_text('▶ Play')
            self.fig.canvas.draw_idle()
        
        threading.Thread(target=_play, daemon=True).start()
    
    def check_answer_button(self, event):
        """Check the answer when button is clicked"""
        text = self.text_input.text
        if not text.strip():
            self.status_text.set_text('⚠️ Please enter an answer first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        # Check if timer was started (i.e., user clicked play)
        if not self.timer_started:
            self.status_text.set_text('⚠️ Please click Play first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        self.check_answer(text)
    
    def check_answer(self, text):
        """Check if the user's answer is correct"""
        if not self.current_item or self.answered:
            return
        
        # Calculate elapsed time
        if self.question_start_time is not None:
            self.question_elapsed_time = time.time() - self.question_start_time
        else:
            self.question_elapsed_time = 0
        
        # Clean up user input - remove spaces, commas
        user_answer = text.strip().replace(' ', '').replace(',', '').replace('-', '')
        
        if not user_answer:
            return
        
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        print(f"\n{'─'*60}")
        print(f"ANSWER SUBMITTED")
        print(f"{'─'*60}")
        print(f"User answer: '{user_answer}'")
        print(f"Correct answer: '{correct_answer}'")
        print(f"Time taken: {self.question_elapsed_time:.2f} seconds")
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        
        # Store result
        result = {
            'question_number': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'correct_tone': correct_answer,
            'user_answer': user_answer,
            'is_correct': is_correct,
            'time_seconds': round(self.question_elapsed_time, 2),
            'audio_file': self.current_item['audio']
        }
        self.results.append(result)
        
        # Show what the user typed
        self.answer_text.set_text(f"Your answer: {user_answer}")
        
        if is_correct:
            # CORRECT
            self.feedback_text.set_text('✓ CORRECT!')
            self.feedback_text.set_color('#27ae60')  # Green
            self.status_text.set_text('Great job!')
            self.status_text.set_color('#27ae60')
            print("✓ CORRECT!")
        else:
            # INCORRECT - show correct answer
            self.feedback_text.set_text(f'✗ INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')  # Red
            self.status_text.set_text('Try again with the next one')
            self.status_text.set_color('#e74c3c')
            print(f"✗ INCORRECT! Correct answer: {correct_answer}")
        
        print(f"{'─'*60}\n")
        self.fig.canvas.draw_idle()
    
    def _save_results_to_file(self):
        """Save all results to a text file"""
        try:
            # Create results directory if it doesn't exist
            script_dir = Path(__file__).parent
            results_dir = script_dir / 'tone_quiz_results'
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"tone_quiz_{timestamp}.txt"
            filepath = results_dir / filename
            
            # Calculate statistics
            total_questions = len(self.results)
            correct_count = sum(1 for r in self.results if r['is_correct'])
            accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
            total_time = sum(r['time_seconds'] for r in self.results)
            avg_time = total_time / total_questions if total_questions > 0 else 0
            
            # Write results to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("MANDARIN TONE INTRODUCTION QUIZ - RESULTS\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Correct Answers: {correct_count}\n")
                f.write(f"Accuracy: {accuracy:.1f}%\n")
                f.write(f"Total Time: {total_time:.2f} seconds\n")
                f.write(f"Average Time per Question: {avg_time:.2f} seconds\n")
                f.write("\n" + "="*70 + "\n\n")
                
                # Write detailed results for each question
                for result in self.results:
                    f.write(f"Question {result['question_number']}/{self.max_questions}\n")
                    f.write(f"{'-'*70}\n")
                    f.write(f"Chinese:       {result['chinese']}\n")
                    f.write(f"Pinyin:        {result['pinyin']}\n")
                    f.write(f"Correct Tone:  {result['correct_tone']}\n")
                    f.write(f"Your Answer:   {result['user_answer']}\n")
                    f.write(f"Result:        {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}\n")
                    f.write(f"Time Taken:    {result['time_seconds']} seconds\n")
                    f.write(f"Audio File:    {result['audio_file']}\n")
                    f.write("\n")
                
                f.write("="*70 + "\n")
                f.write("END OF RESULTS\n")
                f.write("="*70 + "\n")
            
            print(f"\n{'='*70}")
            print(f"✅ RESULTS SAVED TO FILE")
            print(f"{'='*70}")
            print(f"Filename: {filename}")
            print(f"Location: {filepath}")
            print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_questions} correct)")
            print(f"Average Time: {avg_time:.2f} seconds per question")
            print(f"{'='*70}\n")
            
            return filepath
            
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")
            return None
    
    def next_word(self, event):
        """Move to next word"""
        self.question_count += 1
        
        if self.question_count >= self.max_questions:
            # Completed 5 questions, save results and start practice
            print(f"\n{'='*60}")
            print(f"COMPLETED {self.max_questions} QUESTIONS!")
            print(f"{'='*60}\n")
            
            # Save results to file
            self._save_results_to_file()
            
            # Start practice
            self._start_practice()
        else:
            # Move to next word
            self._select_random_item()
    
    def _start_practice(self):
        """Launch the main practice application"""
        print("\n" + "="*60)
        print("STARTING MAIN PRACTICE SESSION")
        print("="*60 + "\n")
        
        sd.stop()
        plt.close(self.fig)
        
    def show(self):
        plt.show()

if __name__ == '__main__':
    # Auto-detect audio path (works on all computers)
    
    print("\n" + "="*60)
    print("MANDARIN TONE INTRODUCTION QUIZ (5 Questions)")
    print("="*60)
    print(f"Script location: {Path(__file__).parent}")
    
    intro = ToneIntroductionQuiz()
    intro.show()
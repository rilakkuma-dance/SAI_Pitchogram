import random
import json
import os
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import sys

class VocabularyPracticeSession:
    def __init__(self, vocab_db):
        self.vocab_db = vocab_db
        self.current_set = None
        self.set_number = 0
        self.practice_history = []
        
    def generate_practice_set(self):
        """Generate a random set of 3 vocabulary words and 2 sentences"""
        self.set_number += 1
        
        # Get available words and sentences
        available_words = list(self.vocab_db['words'].keys())
        available_sentences = self.vocab_db['sentences'].copy()
        
        # Randomly select 3 words
        selected_words = random.sample(available_words, min(3, len(available_words)))
        
        # Randomly select 2 sentences
        selected_sentences = random.sample(available_sentences, min(2, len(available_sentences)))
        
        # Build practice set
        practice_set = {
            'set_number': self.set_number,
            'words': [],
            'sentences': []
        }
        
        # Add word details
        for word in selected_words:
            word_info = self.vocab_db['words'][word]
            practice_set['words'].append({
                'character': word,
                'pinyin': word_info['pinyin'],
                'english': word_info['english'],
                'phonemes': word_info['phonemes'],
                'tone': word_info['tone']
            })
        
        # Add sentence details
        for sentence in selected_sentences:
            practice_set['sentences'].append({
                'id': sentence['id'],
                'mandarin': sentence['mandarin'],
                'pinyin': sentence['pinyin'],
                'english': sentence['english'],
                'phonemes': sentence['phonemes']
            })
        
        self.current_set = practice_set
        self.practice_history.append(practice_set)
        
        return practice_set
    
    def get_current_item(self, item_index):
        """Get specific item from current set (0-4: 3 words + 2 sentences)"""
        if not self.current_set:
            return None
            
        if item_index < 3:
            # Word items
            if item_index < len(self.current_set['words']):
                return self.current_set['words'][item_index]
        else:
            # Sentence items
            sentence_index = item_index - 3
            if sentence_index < len(self.current_set['sentences']):
                return self.current_set['sentences'][sentence_index]
        
        return None
    
    def print_current_set(self):
        """Print current practice set for console display"""
        if not self.current_set:
            print("No practice set generated yet")
            return
        
        print(f"\n{'='*60}")
        print(f"PRACTICE SET #{self.current_set['set_number']}")
        print(f"{'='*60}")
        
        print("\n📚 VOCABULARY WORDS:")
        for i, word in enumerate(self.current_set['words'], 1):
            tone_str = str(word['tone']) if isinstance(word['tone'], int) else ','.join(map(str, word['tone']))
            print(f"  {i}. {word['character']} ({word['pinyin']}) - {word['english']}")
            print(f"     Phonemes: {word['phonemes']} | Tone: {tone_str}")
        
        print("\n📝 SENTENCES:")
        for i, sentence in enumerate(self.current_set['sentences'], 1):
            print(f"  {i}. {sentence['mandarin']}")
            print(f"     {sentence['pinyin']}")
            print(f"     '{sentence['english']}'")
            print(f"     Phonemes: {sentence['phonemes']}")
        
        print(f"{'='*60}\n")

def add_practice_mode_to_visualization(sai_vis_class):
    """Extend SAI visualization with practice mode"""
    
    original_init = sai_vis_class.__init__
    
    def new_init(self, *args, **kwargs):
        import sys
        main_module = sys.modules['__main__']
        PracticeModeBrowserFeedback = main_module.PracticeModeBrowserFeedback
        VOCAB_DB = main_module.VOCAB_DB
        PhonemeAnalyzer = main_module.PhonemeAnalyzer
        
        # EXTRACT audio_manager from kwargs BEFORE calling original_init
        audio_manager = kwargs.get('audio_manager', None)  # Changed from pop to get
        
        # Initialize practice session attributes BEFORE calling original_init
        self.practice_session = VocabularyPracticeSession(VOCAB_DB)
        self.current_practice_set = None
        self.current_item_index = 0
        self.current_practice_item = None
        self.PhonemeAnalyzer = PhonemeAnalyzer
        self.audio_manager = audio_manager  # Store it here

        original_webview_setting = kwargs.get('use_webview', True)
        kwargs['use_webview'] = False
        
        # NOW call the original init (it will also get audio_manager since we used 'get' not 'pop')
        original_init(self, *args, **kwargs)
        
        # Now enable practice mode feedback if it was requested
        if original_webview_setting:
            self.use_browser_feedback = True
            self.feedback_webview = PracticeModeBrowserFeedback()
            self.feedback_webview.start()
            print("Practice mode browser feedback started")
        
        self._needs_practice_init = True

    def generate_new_practice_set(self):
        """Generate and display new practice set"""
        self.current_practice_set = self.practice_session.generate_practice_set()
        self.current_item_index = 0
        self.practice_session.print_current_set()
        self.set_current_practice_item(0)
    
    def set_current_practice_item(self, item_index):
        """Set current item being practiced"""
        print(f"DEBUG: set_current_practice_item called, audio_manager exists: {self.audio_manager is not None}")
        
        import sys
        main_module = sys.modules['__main__']
        PhonemeAnalyzer = main_module.PhonemeAnalyzer
        
        self.current_item_index = item_index
        item = self.practice_session.get_current_item(item_index)
        
        if item:
            self.current_practice_item = item
            self.target_phonemes = item['phonemes']
            self.wav2vec2_handler.target_phonemes = item['phonemes']
            self.wav2vec2_handler.phoneme_analyzer = PhonemeAnalyzer(item['phonemes'])
            
            # Load audio file - handle both words and sentences
            if self.audio_manager:
                voice_type = 'women' if self.voice_selector.current_voice == 'female' else 'men'
                
                if 'character' in item:
                    # This is a word
                    word = item['character']
                    audio_path = self.audio_manager.get_audio_for_word(word, voice_type, 0)
                    
                    if audio_path:
                        print(f"Loading audio for word '{word}': {os.path.basename(audio_path)}")
                        self._switch_audio_file(audio_path)
                    else:
                        print(f"Warning: No audio file found for word '{word}'")
                
                elif 'id' in item:
                    # This is a sentence - use sentence ID
                    sentence_id = item['id']
                    audio_path = self.audio_manager.get_audio_for_sentence(sentence_id, voice_type, 0)
                    
                    if audio_path:
                        print(f"Loading audio for sentence ID {sentence_id}: {os.path.basename(audio_path)}")
                        self._switch_audio_file(audio_path)
                    else:
                        print(f"Warning: No audio file found for sentence ID {sentence_id}")
            
            if 'character' in item:
                print(f"DEBUG: Setting reference to {item['pinyin']} - {item['english']}")
                self.set_reference_text(item['phonemes'], item['pinyin'], item['english'])            
            else:
                self.set_reference_text(item['phonemes'], item['pinyin'], item['english'])
            
            print(f"\n>> Now practicing: {item.get('character', item.get('mandarin', ''))}")
            print(f">> Target: {item['pinyin']} - {item['english']}")
            print(f">> Phonemes: {item['phonemes']}\n")
            
            self.update_browser_with_practice_context()
        
    def next_practice_item(self, event=None):
        """Move to next item in set"""
        self.current_item_index += 1
        if self.current_item_index >= 5:
            print("\n✅ Practice set complete! Generating new set...")
            self.generate_new_practice_set()
        else:
            self.set_current_practice_item(self.current_item_index)
    
    def update_browser_with_practice_context(self):
        """Update browser feedback with practice context"""
        print(f"DEBUG: update_browser_with_practice_context - has item: {hasattr(self, 'current_practice_item')}")
        if hasattr(self, 'current_practice_item') and self.current_practice_item:
            print(f"DEBUG: current_practice_item is: {self.current_practice_item.get('character', 'NO CHAR')}")
        
        if not (self.use_browser_feedback and self.feedback_webview and self.current_practice_set):
            print("DEBUG: Missing requirements for browser update")
            return
        
        feedback_data = {
            'overall_score': 0,
            'target': [],
            'yours': [],
            'practice_set': {
                'set_number': self.current_practice_set['set_number'],
                'current_item': f"Item {self.current_item_index + 1} of 5",
                'words': self.current_practice_set['words'],
                'sentences': self.current_practice_set['sentences']
            }
        }
        
        if hasattr(self, 'current_practice_item') and self.current_practice_item:
            item = self.current_practice_item
            
            if 'character' in item:
                characters = list(item['character'])
                tones = item['tone'] if isinstance(item['tone'], list) else [item['tone']] * len(characters)
                phoneme_list = list(item['phonemes'])
                phonemes_per_char = max(1, len(phoneme_list) // len(characters))
                
                for idx, char in enumerate(characters):
                    start = idx * phonemes_per_char
                    end = start + phonemes_per_char
                    char_phonemes = phoneme_list[start:end] if start < len(phoneme_list) else []
                    
                    feedback_data['target'].append({
                        'character': char,
                        'pinyin': item['pinyin'],
                        'tone': tones[idx] if idx < len(tones) else 1,
                        'phonemes': char_phonemes,
                        'phoneme_scores': ['perfect'] * len(char_phonemes)
                    })

        print(f"DEBUG: Sending to browser - targets: {len(feedback_data['target'])}, yours: {len(feedback_data['yours'])}")
        if feedback_data['target']:
            print(f"DEBUG: First target item: {feedback_data['target'][0]}")
        
        if (hasattr(self, 'tone_analysis_results') and self.tone_analysis_results and
            hasattr(self.wav2vec2_handler, 'analysis_results') and 
            self.wav2vec2_handler.analysis_results):
            
            phoneme_results = {
                'overall_score': self.wav2vec2_handler.overall_score,
                'analysis_results': self.wav2vec2_handler.analysis_results
            }
            
            full_feedback = self.create_webview_feedback_data(
                self.tone_analysis_results,
                phoneme_results
            )
            
            feedback_data['overall_score'] = full_feedback['overall_score']
            feedback_data['yours'] = full_feedback['yours']
            feedback_data['target'] = full_feedback['target']
        
        self.feedback_webview.update_practice_feedback(
            self.current_practice_set,
            self.current_item_index,
            feedback_data
        )
    
    # Attach all methods to the class
    sai_vis_class.generate_new_practice_set = generate_new_practice_set
    sai_vis_class.set_current_practice_item = set_current_practice_item
    sai_vis_class.next_practice_item = next_practice_item
    sai_vis_class.update_browser_with_practice_context = update_browser_with_practice_context
    sai_vis_class.__init__ = new_init
    
    return sai_vis_class

def add_practice_button_to_setup(original_setup_method):
    """Decorator to add practice controls to visualization"""
    def enhanced_setup(self):
        # Call original setup first
        original_setup_method(self)
        
        # Add "Next Item" button
        button_width = 0.10
        button_height = 0.04
        next_x = 0.65
        next_y = 0.07
        
        self.ax_next_item = plt.axes([next_x, next_y, button_width, button_height])
        self.btn_next_item = Button(self.ax_next_item, 'Next Item →', 
                                    color='lightgreen', hovercolor='green')
        self.btn_next_item.on_clicked(self.next_practice_item)
        
        # Add "New Set" button
        self.ax_new_set = plt.axes([next_x + button_width + 0.01, next_y, button_width, button_height])
        self.btn_new_set = Button(self.ax_new_set, 'New Set ↻', 
                                  color='lightyellow', hovercolor='yellow')
        self.btn_new_set.on_clicked(lambda e: self.generate_new_practice_set())
        
        # Add item counter display
        self.item_counter_display = self.ax_controls.text(
            0.73, 0.05, f'Item 1/5 | Set #{self.practice_session.set_number}',
            transform=self.ax_controls.transAxes,
            fontsize=10, color='cyan', weight='bold', horizontalalignment='center'
        )
        
        # CRITICAL: Generate practice set NOW after buttons are created
        print("DEBUG: About to generate practice set...")
        self.generate_new_practice_set()
        print("DEBUG: Practice set generation complete")
    
    return enhanced_setup

# Step 2: Override tone callback to update browser with practice context
def enhanced_tone_callback(self, audio):
    """Enhanced callback that includes practice context"""
    # Original tone analysis
    if self.tone_analyzer is not None:
        try:
            self.tone_analysis_results = self.tone_analyzer.predict_tone(audio)
            print(f"Tone analysis: {self.tone_analysis_results}")
        except Exception as e:
            print(f"Tone analysis error: {e}")
            self.tone_analysis_results = None
    
    # Update browser with practice context
    self.update_browser_with_practice_context()


# Step 3: Update item counter display
def update_item_counter_display(self):
    """Update the practice item counter"""
    if hasattr(self, 'item_counter_display') and hasattr(self, 'practice_session'):
        counter_text = f'Item {self.current_item_index + 1}/5 | Set #{self.practice_session.set_number}'
        self.item_counter_display.set_text(counter_text)


def enhance_visualization_with_practice(viz_class):
    """Apply all practice mode enhancements"""
    enhanced_class = add_practice_mode_to_visualization(viz_class)
    
    original_setup = enhanced_class._setup_dual_visualization
    enhanced_class._setup_dual_visualization = add_practice_button_to_setup(original_setup)
    
    enhanced_class.tone_analysis_callback = enhanced_tone_callback
    enhanced_class.update_item_counter_display = update_item_counter_display
    
    original_next = enhanced_class.next_practice_item
    def next_with_display_update(self, event=None):
        original_next(self, event)
        self.update_item_counter_display()
    enhanced_class.next_practice_item = next_with_display_update
    
    return enhanced_class

print("✓ Practice mode module loaded")
import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from matplotlib.widgets import Button
import threading
import queue
import librosa
import argparse
import os
import sounddevice as sd
import wave
from datetime import datetime
import speech_recognition as sr
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

from feedback_ui import enhance_visualization_with_practice 
# ---------------- Visualization Handler ----------------
from modules.visualization_handler import VisualizationHandler, SAIParams

# ---------------- Phoneme Alignment and Feedback System ----------------
from modules.phoneme_handler import PhonemeHandler, PhonemeAnalyzer

# ---------------- Audio Recorder ----------------
from modules.recorder import AudioRecorder

# ---------------- Tone Analyzer ----------------
from modules.tone_detection_sentence import Wav2Vec2PypinyinToneClassifier as ToneAnalyzerSentence
from modules.tone_detection_word import ToneClassifierTester as ToneAnalyzerWord

# Load vocabulary database
def load_mandarin_vocab(filename='mandarin_vocab.json'):
    """Load Mandarin vocabulary from JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['words'])} words and {len(data['sentences'])} sentences")
        return data
    except FileNotFoundError:
        print(f"Vocab file {filename} not found, using defaults")
        return {"words": {}, "sentences": []}
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return {"words": {}, "sentences": []}

# Load the database
VOCAB_DB = load_mandarin_vocab()

def get_word_info(word):
    """Get information about a Chinese word"""
    return VOCAB_DB['words'].get(word, None)

def get_sentence_by_id(sentence_id):
    """Get a sentence by its ID"""
    for sentence in VOCAB_DB['sentences']:
        if sentence['id'] == sentence_id:
            return sentence
    return None

def get_random_sentence():
    """Get a random sentence for practice"""
    import random
    if VOCAB_DB['sentences']:
        return random.choice(VOCAB_DB['sentences'])
    return None

def list_all_sentences():
    """List all available sentences"""
    return [(s['id'], s['mandarin'], s['english']) for s in VOCAB_DB['sentences']]

def set_target_word(word):
    """Set target word for practice"""
    info = get_word_info(word)
    if info:
        return {
            'character': word,
            'pinyin': info['pinyin'],
            'phonemes': info['phonemes'],
            'tone': info['tone'],
            'english': info['english']
        }
    return None

# Font setup
def get_font_path():
    """Get font path relative to script location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "fonts", "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "DoulosSIL-7.000", "DoulosSIL-Regular.ttf"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Initialize font
font_path = get_font_path()

if font_path:
    font_prop = fm.FontProperties(fname=font_path, size=16)
    print(f"Using Doulos SIL font: {font_path}")
else:
    font_prop = fm.FontProperties(family='Times New Roman', size=16)
    print("Doulos SIL not found, using Times New Roman fallback")

# Wav2Vec2 imports
try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
    WAV2VEC2_AVAILABLE = True
except ImportError:
    print("Warning: torch/transformers not found. Install with: pip install torch torchaudio transformers")
    WAV2VEC2_AVAILABLE = False

# JAX/CARFAC imports
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    import carfac.jax.carfac as carfac
    from carfac.np.carfac import CarParams
    import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Using simplified visualization.")
    JAX_AVAILABLE = False

class BrowserFeedback:
    """Browser-based feedback using auto-refreshing JSON data"""
    
    def __init__(self, port=8765):
        self.port = port
        self.ready = False
        self.html_file = 'feedback_ui.html'
        self.data_file = 'feedback_data.json'
        self.server = None
        
        # Create HTML with auto-refresh
        self._create_html_with_autoreload()
        
        # Initialize empty data
        self.update_feedback({
            'overall_score': 0,
            'target': [],
            'yours': []
        })
    
    def _create_html_with_autoreload(self):
        """Create HTML that auto-reloads data from JSON"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pronunciation Feedback</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .score-card {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .overall-score {
            font-size: 4rem;
            font-weight: bold;
            background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .score-label { font-size: 1.2rem; opacity: 0.8; letter-spacing: 2px; }
        .feedback-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .row-label {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .target-color { color: #60efff; }
        .yours-color { color: #ffb347; }
        .syllables-container {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .syllable-card {
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            min-width: 140px;
            transition: transform 0.2s;
        }
        .syllable-card:hover { transform: translateY(-5px); }
        .character { font-size: 3rem; text-align: center; margin-bottom: 10px; color: #60efff; }
        .tone-visual {
            height: 50px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .tone-line { stroke: #00ff87; stroke-width: 3; fill: none; filter: drop-shadow(0 0 5px #00ff87); }
        .pinyin {
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 12px;
            font-weight: 600;
            color: #60efff;
        }
        .phonemes { display: flex; gap: 4px; justify-content: center; flex-wrap: wrap; }
        .phoneme {
            padding: 6px 10px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .perfect { background: #00ff87; color: #000; }
        .good { background: #4ade80; color: #000; }
        .close { background: #fbbf24; color: #000; }
        .poor { background: #fb923c; color: #000; }
        .wrong { background: #ef4444; color: #fff; }
        .missing { background: #6b7280; color: #fff; opacity: 0.5; }
        .confidence {
            text-align: center;
            margin-top: 10px;
            font-size: 0.85rem;
            opacity: 0.8;
        }
        .confidence-value {
            display: inline-block;
            background: rgba(0, 255, 135, 0.2);
            padding: 3px 10px;
            border-radius: 12px;
            margin-top: 5px;
        }
        .vs-separator { text-align: center; font-size: 2rem; margin: 20px 0; opacity: 0.3; }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
        }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
        .legend-dot { width: 16px; height: 16px; border-radius: 50%; }
        .status { position: fixed; top: 10px; right: 10px; padding: 10px 20px; background: rgba(0,0,0,0.8); border-radius: 10px; font-size: 0.9rem; }
        .status.active { border: 2px solid #00ff87; }
    </style>
</head>
<body>
    <div class="status" id="status">Waiting for data...</div>
    <div class="container">
        <div class="score-card">
            <div class="overall-score" id="overallScore">--</div>
            <div class="score-label">OVERALL PRONUNCIATION SCORE</div>
        </div>
        <div class="feedback-section">
            <div class="row-label target-color">🎯 TARGET: <span id="targetWord">Loading...</span></div>
            <div class="syllables-container" id="targetSyllables"></div>
        </div>
        <div class="vs-separator">↓</div>
        <div class="feedback-section">
            <div class="row-label yours-color">🗣️ YOUR PRONUNCIATION:</div>
            <div class="syllables-container" id="yourSyllables"></div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot perfect"></div><span>Perfect</span></div>
            <div class="legend-item"><div class="legend-dot good"></div><span>Good</span></div>
            <div class="legend-item"><div class="legend-dot close"></div><span>Close</span></div>
            <div class="legend-item"><div class="legend-dot poor"></div><span>Poor</span></div>
            <div class="legend-item"><div class="legend-dot wrong"></div><span>Wrong</span></div>
            <div class="legend-item"><div class="legend-dot missing"></div><span>Missing</span></div>
        </div>
    </div>
    <script>
        const toneShapes = {
            1: '<svg width="100" height="50" viewBox="0 0 100 50"><path class="tone-line" d="M 10 15 L 90 15"/></svg>',
            2: '<svg width="100" height="50" viewBox="0 0 100 50"><path class="tone-line" d="M 10 40 Q 50 25 90 10"/></svg>',
            3: '<svg width="100" height="50" viewBox="0 0 100 50"><path class="tone-line" d="M 10 25 Q 25 40 50 45 Q 75 40 90 15"/></svg>',
            4: '<svg width="100" height="50" viewBox="0 0 100 50"><path class="tone-line" d="M 10 10 L 90 45"/></svg>',
            5: '<svg width="100" height="50" viewBox="0 0 100 50"><path class="tone-line" d="M 10 25 L 90 25" stroke-dasharray="5,5"/></svg>'
        };

        function createSyllableCard(data, isTarget = true) {
            let content = `<div class="character">${data.character}</div>`;
            if (data.tone) {
                content += `<div class="tone-visual">${toneShapes[data.tone] || ''}</div>`;
            } else {
                content += `<div class="tone-visual" style="opacity: 0.3;">❓</div>`;
            }
            content += `<div class="pinyin">${data.pinyin}${data.tone || '?'}</div><div class="phonemes">`;
            data.phonemes.forEach((phoneme, idx) => {
                const scoreClass = data.phoneme_scores[idx] || 'perfect';
                content += `<span class="phoneme ${scoreClass}">${phoneme}</span>`;
            });
            content += `</div>`;
            if (!isTarget && data.confidence !== undefined) {
                content += `<div class="confidence">Confidence: <span class="confidence-value">${(data.confidence * 100).toFixed(0)}%</span></div>`;
            }
            return content;
        }

        function updateFeedback(data) {
            document.getElementById('overallScore').textContent = `${Math.round(data.overall_score * 100)}%`;
            
            const targetContainer = document.getElementById('targetSyllables');
            targetContainer.innerHTML = '';
            data.target.forEach(syllable => {
                const card = document.createElement('div');
                card.className = 'syllable-card';
                card.innerHTML = createSyllableCard(syllable, true);
                targetContainer.appendChild(card);
            });
            
            const yourContainer = document.getElementById('yourSyllables');
            yourContainer.innerHTML = '';
            data.yours.forEach(syllable => {
                const card = document.createElement('div');
                card.className = 'syllable-card';
                card.innerHTML = createSyllableCard(syllable, false);
                yourContainer.appendChild(card);
            });
        }

        let lastUpdate = 0;
        setInterval(() => {
            fetch('feedback_data.json?' + Date.now())
                .then(r => r.json())
                .then(data => {
                    if (data && data.overall_score !== undefined && data.updated_at !== lastUpdate) {
                        updateFeedback(data);
                        lastUpdate = data.updated_at;
                        const status = document.getElementById('status');
                        status.textContent = 'Connected ✓';
                        status.className = 'status active';
                    }
                })
                .catch(e => {
                    const status = document.getElementById('status');
                    status.textContent = 'Waiting for data...';
                    status.className = 'status';
                });
        }, 500);
    </script>
</body>
</html>'''
        
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Created HTML file: {self.html_file}")
    
    def start(self):
        """Start HTTP server and open browser"""
        def run_server():
            class Handler(SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass  # Suppress server logs
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            self.server = HTTPServer(('localhost', self.port), Handler)
            self.ready = True
            print(f"Feedback server started at http://localhost:{self.port}")
            self.server.serve_forever()
        
        threading.Thread(target=run_server, daemon=True).start()
        time.sleep(1)
        
        url = f'http://localhost:{self.port}/{self.html_file}'
        webbrowser.open(url)
        print(f"Browser opened: {url}")
    
    def update_feedback(self, feedback_data):
        """Save feedback data to JSON file"""
        try:
            feedback_data['updated_at'] = time.time()

            if hasattr(self, 'script_dir'):
                data_file_path = os.path.join(self.script_dir, self.data_file)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                data_file_path = os.path.join(script_dir, self.data_file)
            
            # Get the absolute path to ensure we write to the correct location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(script_dir, self.data_file)
            
            print(f"DEBUG: Writing to {data_file_path}: targets={len(feedback_data.get('target', []))}, yours={len(feedback_data.get('yours', []))}")
            
            with open(data_file_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            
            print(f"DEBUG: Successfully wrote feedback_data.json")
        except Exception as e:
            print(f"Error saving feedback: {e}")


class PracticeModeBrowserFeedback(BrowserFeedback):
    """Enhanced browser feedback with practice set display"""
    
    def __init__(self, port=8765):
        # Don't call super().__init__ - do it manually
        self.port = port
        self.ready = False
        self.html_file = 'feedback_ui.html'
        self.data_file = 'feedback_data.json'
        self.server = None

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create practice mode HTML
        self._create_practice_mode_html()
        
        # Initialize with empty data
        self.update_feedback({
            'overall_score': 0,
            'target': [],
            'yours': []
        })
    
    def _create_practice_mode_html(self):
        """Create enhanced HTML with practice set display"""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mandarin Practice Mode</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 10px;
            height: 100vh;
            overflow: hidden;
        }
        .container { max-width: 1200px; margin: 0 auto; height: 100%; display: flex; flex-direction: column; }
        
        .practice-header {
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(0, 255, 135, 0.5);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            text-align: center;
        }
        .set-title { font-size: 1.5rem; color: #00ff87; margin-bottom: 5px; }
        .current-item { font-size: 1rem; color: #60efff; }
        
        .score-section {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 12px;
            margin-bottom: 12px;
            text-align: center;
        }
        .overall-score {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        
        .feedback-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            flex-grow: 1;
        }
        .feedback-column {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px;
        }
        .column-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        .target-title { color: #60efff; }
        .yours-title { color: #ffb347; }
        
        .syllables-row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .syllable-box {
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            min-width: 100px;
        }
        .syll-char { font-size: 2rem; text-align: center; color: #60efff; }
        .syll-pinyin { text-align: center; font-size: 0.9rem; color: #60efff; margin: 5px 0; }
        .phonemes { display: flex; gap: 3px; justify-content: center; flex-wrap: wrap; margin-top: 8px; }
        .phoneme {
            padding: 4px 6px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            font-weight: bold;
        }
        .perfect { background: #00ff87; color: #000; }
        .good { background: #4ade80; color: #000; }
        .close { background: #fbbf24; color: #000; }
        .poor { background: #fb923c; color: #000; }
        .wrong { background: #ef4444; color: #fff; }
        .missing { background: #6b7280; color: #fff; opacity: 0.5; }
        
        .status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 15px;
            background: rgba(0,0,0,0.8);
            border-radius: 8px;
            font-size: 0.8rem;
            border: 2px solid #666;
        }
        .status.active { border-color: #00ff87; }
        .tone-visual {
            height: 25px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div class="status" id="status">Waiting for data...</div>
    <div class="container">
        <div class="practice-header">
            <div class="set-title" id="setTitle">Practice Set #1</div>
            <div class="current-item" id="currentItem">Item 1 of 5</div>
        </div>

        <div class="score-section">
            <div class="overall-score" id="overallScore">--</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">PRONUNCIATION SCORE</div>
        </div>
        
        <div class="feedback-grid">
            <div class="feedback-column">
                <div class="column-title target-title">🎯 TARGET</div>
                <div class="syllables-row" id="targetSyllables"></div>
            </div>
            <div class="feedback-column">
                <div class="column-title yours-title">🗣️ YOUR PRONUNCIATION</div>
                <div class="syllables-row" id="yourSyllables"></div>
            </div>
        </div>
    </div>
    
    <script>
        const toneShapes = {
            1: '<svg width="60" height="25" viewBox="0 0 100 50"><path class="tone-line" d="M 10 15 L 90 15" stroke="#00ff87" stroke-width="2" fill="none"/></svg>',
            2: '<svg width="60" height="25" viewBox="0 0 100 50"><path class="tone-line" d="M 10 40 Q 50 25 90 10" stroke="#00ff87" stroke-width="2" fill="none"/></svg>',
            3: '<svg width="60" height="25" viewBox="0 0 100 50"><path class="tone-line" d="M 10 25 Q 25 40 50 45 Q 75 40 90 15" stroke="#00ff87" stroke-width="2" fill="none"/></svg>',
            4: '<svg width="60" height="25" viewBox="0 0 100 50"><path class="tone-line" d="M 10 10 L 90 45" stroke="#00ff87" stroke-width="2" fill="none"/></svg>',
            5: '<svg width="60" height="25" viewBox="0 0 100 50"><path class="tone-line" d="M 10 25 L 90 25" stroke="#00ff87" stroke-width="2" stroke-dasharray="5,5" fill="none"/></svg>'
        };
        
        function createSyllableBox(data, isTarget) {
            const box = document.createElement('div');
            box.className = 'syllable-box';
            
            let content = `<div class="syll-char">${data.character}</div>`;
            
            if (data.tone) {
                content += `<div class="tone-visual">${toneShapes[data.tone] || ''}</div>`;
            } else {
                content += `<div class="tone-visual" style="opacity: 0.3;">❓</div>`;
            }
            
            content += `<div class="syll-pinyin">${data.pinyin}${data.tone || ''}</div>`;
            content += '<div class="phonemes">';
            
            data.phonemes.forEach((phoneme, idx) => {
                const scoreClass = data.phoneme_scores[idx] || 'perfect';
                content += `<span class="phoneme ${scoreClass}">${phoneme}</span>`;
            });
            
            content += '</div>';
            box.innerHTML = content;
            return box;
        }
        
        function updateDisplay(data) {
            console.log('updateDisplay called with:', data);
            
            if (!data) return;
            
            if (data.practice_set) {
                document.getElementById('setTitle').textContent = 
                    `Practice Set #${data.practice_set.set_number}`;
                document.getElementById('currentItem').textContent = 
                    data.practice_set.current_item || 'Item 1 of 5';
            }
            
            document.getElementById('overallScore').textContent = 
                data.overall_score ? `${Math.round(data.overall_score * 100)}%` : '--';
            
            const targetContainer = document.getElementById('targetSyllables');
            targetContainer.innerHTML = '';
            if (data.target && data.target.length > 0) {
                data.target.forEach(syll => {
                    targetContainer.appendChild(createSyllableBox(syll, true));
                });
            }
            
            const yoursContainer = document.getElementById('yourSyllables');
            yoursContainer.innerHTML = '';
            if (data.yours && data.yours.length > 0) {
                data.yours.forEach(syll => {
                    yoursContainer.appendChild(createSyllableBox(syll, false));
                });
            }
        }
        
        let lastUpdate = 0;
        setInterval(() => {
            fetch('feedback_data.json?' + Date.now())
                .then(r => r.json())
                .then(data => {
                    if (data) {
                        updateDisplay(data);
                        lastUpdate = data.updated_at;
                        const status = document.getElementById('status');
                        status.textContent = 'Connected ✓';
                        status.className = 'status active';
                    }
                })
                .catch(e => {
                    console.error('Fetch error:', e);
                    const status = document.getElementById('status');
                    status.textContent = 'Waiting...';
                    status.className = 'status';
                });
        }, 500);
    </script>
</body>
</html>'''
    
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Practice mode HTML created: {self.html_file}")

            # ADD THIS DEBUG LINE
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.html_file)
        print(f"DEBUG: Creating HTML at: {filepath}")
        
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Practice mode HTML created: {self.html_file}")
    
    def update_practice_feedback(self, practice_set, current_item_index, feedback_data):
        """Update feedback with practice set context"""
        feedback_data['practice_set'] = {
            'set_number': practice_set['set_number'],
            'current_item': f"Item {current_item_index + 1} of 5",
            'words': practice_set['words'],
            'sentences': practice_set['sentences']
        }
        print(f"DEBUG: update_practice_feedback called with {len(feedback_data.get('target', []))} targets")
        self.update_feedback(feedback_data)
# ---------------- Chunk-based Wav2Vec2 Phoneme Handler ----------------
class SimpleWav2Vec2Handler:
    """Wav2Vec2 phoneme recognition handler - records all audio then processes once"""
    
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft", sample_rate=16000, target_phonemes="ɕiɛɕiɛ"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.enabled = True
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
        # Audio buffer for mic recording
        self.audio_buffer = []
        self.is_recording = False
        self.is_processing = False
        self.result = None

        # Phoneme analyzer setup
        self.target_phonemes = target_phonemes
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)
        
        # Add phoneme analysis results storage
        self.analysis_results = None
        self.overall_score = 0.0

        # Callback system
        self.callbacks = []

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

        # Load model immediately
        if not self.load_model():
            print("Failed to load Wav2Vec2 phoneme model. Handler disabled.")
            self.enabled = False

    def load_model(self):
        try:
            print(f"Loading Wav2Vec2 {self.model_name}...")
            
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            print(f"Model loaded: {type(self.model)}")
            print(f"Feature extractor loaded: {type(self.feature_extractor)}")
            
            self.tokenizer = None
            
            try:
                self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                print(f"CTC Tokenizer loaded: {type(self.tokenizer)}")
                
                if isinstance(self.tokenizer, bool):
                    print("CTC tokenizer returned boolean, trying alternatives...")
                    self.tokenizer = None
                else:
                    test_result = self.tokenizer.decode([1, 2, 3], skip_special_tokens=True)
                    print(f"CTC tokenizer test: '{test_result}'")
                    
            except Exception as e:
                print(f"CTC tokenizer failed: {e}")
                self.tokenizer = None
            
            if self.tokenizer is None or isinstance(self.tokenizer, bool):
                print("Trying alternative working model...")
                
                working_model = "facebook/wav2vec2-base-960h"
                try:
                    print(f"Loading working model: {working_model}")
                    self.model = Wav2Vec2ForCTC.from_pretrained(working_model)
                    self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(working_model)
                    self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(working_model)
                    self.model_name = working_model
                    
                    print(f"Alternative model loaded successfully")
                    print(f"Alternative tokenizer: {type(self.tokenizer)}")
                    
                    if not isinstance(self.tokenizer, bool):
                        test_result = self.tokenizer.decode([1, 2, 3], skip_special_tokens=True)
                        print(f"Alternative tokenizer test: '{test_result}'")
                    else:
                        print("Alternative tokenizer is also boolean!")
                        return False
                        
                except Exception as e:
                    print(f"Alternative model failed: {e}")
                    return False
            
            if self.tokenizer is None or isinstance(self.tokenizer, bool):
                print("All tokenizer loading approaches failed")
                return False
            
            if not all([self.model, self.feature_extractor, self.tokenizer]):
                print("Missing required components")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def record(self, duration=3):
        """Record audio from microphone"""
        if not self.enabled:
            print("Handler not enabled")
            return None
        
        print(f"Recording {duration} seconds from microphone...")
        self.audio_buffer = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = self.audio_buffer.flatten()
        print("Recording finished")
        return audio

    def get_current_status(self):
        """Simple 3-state status"""
        if not self.enabled:
            return "Wav2Vec2 not available", "red"
        
        if self.is_recording:
            return "Recording...", "yellow"
        
        if self.is_processing:
            return "Processing phonemes...", "orange"
        
        if self.result is not None:
            if self.result == "no_audio":
                return "No audio detected", "gray"
            else:
                return f"Transcription: {self.result[:20]}{'...' if len(self.result) > 20 else ''}", "black"
        
        return "Ready to start", "blue"

    def start_recording(self):
        """Start recording"""
        if not self.enabled or self.is_recording:
            return False
        
        if self.model is None:
            print("Error: Model not loaded")
            return False
        
        self.is_recording = True
        self.is_processing = False
        self.result = None
        self.audio_data = []
        
        threading.Thread(target=self._record_audio, daemon=True).start()
        return True

    def stop_recording(self):
        """Stop recording and process"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.is_processing = True
        
        threading.Thread(target=self._process_audio, daemon=True).start()

    def get_current_result(self):
        """Get the latest transcription result"""
        return self.result if self.result else "no_audio"

    def register_callback(self, callback, *args):
        """Register a callback function to be called on audio processing completion"""
        self.callbacks.append((callback, args))

    def run_callbacks(self, complete_audio):
        """Run all registered callbacks"""
        for callback, args in self.callbacks:
            try:
                callback(complete_audio, *args)
            except Exception as e:
                print(f"Callback error: {e}")

    def _record_audio(self):
        """Record audio continuously"""
        try:
            with self.microphone as source:
                stream = source.stream
                while self.is_recording:
                    audio_chunk = stream.read(source.CHUNK)
                    audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_data.append(audio_np)

        except Exception as e:
            print(f"Recording error: {e}")

    def _process_audio(self):
        """Process all recorded audio with Wav2Vec2"""
        try:
            if not self.audio_data:
                self.result = "no_audio"
                self.is_processing = False
                return
            
            complete_audio = np.concatenate(self.audio_data)
            
            print(f"Processing with manual components")
            
            try:
                inputs = self.feature_extractor(
                    complete_audio, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                print(f"Input shape: {inputs.input_values.shape}")
                
                with torch.no_grad():
                    logits = self.model(inputs.input_values).logits
                
                print(f"Logits shape: {logits.shape}")
                
                predicted_ids = torch.argmax(logits, dim=-1)
                
                transcription = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                print(f"Raw transcription: '{transcription}'")
                
                cleaned_transcription = transcription.strip()
                
                if cleaned_transcription and len(cleaned_transcription.strip()) > 0:
                    self.result = cleaned_transcription
                    
                    try:
                        self.analysis_results, self.overall_score = self.phoneme_analyzer.analyze_pronunciation(
                            cleaned_transcription, self.target_phonemes
                        )
                        print(f"Phoneme analysis completed. Overall score: {self.overall_score:.2f}")
                    except Exception as analysis_error:
                        print(f"Phoneme analysis error: {analysis_error}")
                        self.analysis_results = None
                        self.overall_score = 0.0
                else:
                    self.result = "no_audio"
                    self.analysis_results = None
                    self.overall_score = 0.0
                    print("No transcription generated")

                self.run_callbacks(complete_audio)
                    
            except Exception as processing_error:
                print(f"Processing error: {processing_error}")
                import traceback
                traceback.print_exc()
                self.result = "processing_error"
                self.analysis_results = None
                self.overall_score = 0.0
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            self.result = "no_audio"
            self.analysis_results = None
            self.overall_score = 0.0
        
        self.is_processing = False

# ---------------- Audio Processor with Fallback ----------------
class AudioProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        if JAX_AVAILABLE:
            try:
                self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
                    carfac.CarfacDesignParameters(fs=fs, n_ears=1)
                )
                self.n_channels = self.hypers.ears[0].car.n_ch
                self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])
                self.use_carfac = True
                print("Using CARFAC audio processing")
            except Exception as e:
                print(f"CARFAC initialization failed: {e}")
        else:
            raise Exception("CARFAC initialization failed")

    def process_chunk(self, audio_chunk):
        if self.use_carfac:
            try:
                if len(audio_chunk.shape) == 1:
                    audio_input = audio_chunk.reshape(-1, 1)
                else:
                    audio_input = audio_chunk
                audio_jax = jnp.array(audio_input, dtype=jnp.float32)
                naps, _, self.state, _, _, _ = self.run_segment_jit(
                    audio_jax, self.hypers, self.weights, self.state, open_loop=False
                )
                return np.array(naps[:, :, 0]).T
            except Exception as e:
                print(f"CARFAC processing error: {e}")

# ---------------- SAI Processor with Fallback ----------------
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
                print("Using SAI processing")
            except Exception as e:
                print(f"SAI initialization failed: {e}")
                self.use_sai = False
        else:
            self.use_sai = False
        
        if not self.use_sai:
            print("Using simple autocorrelation")
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception as e:
                print(f"SAI processing error: {e}")
                return self._simple_sai(nap_output)
        else:
            return self._simple_sai(nap_output)
    
    def _simple_sai(self, nap_output):
        sai_output = np.zeros((self.sai_params.num_channels, self.sai_params.sai_width))
        
        for ch in range(min(nap_output.shape[0], self.sai_params.num_channels)):
            if nap_output.shape[1] > 0:
                channel_data = nap_output[ch, :]
                for lag in range(min(len(channel_data), self.sai_params.sai_width)):
                    if len(channel_data) > lag:
                        start_idx = max(0, len(channel_data) - lag - 10)
                        end_idx = len(channel_data) - lag
                        if end_idx > start_idx:
                            sai_output[ch, lag] = np.mean(channel_data[start_idx:end_idx])
        
        return sai_output

# ---------------- Waveform Buffer Class ----------------
class WaveformBuffer:
    """Circular buffer for storing waveform data for display"""
    def __init__(self, size=8000):
        self.size = size
        self.buffer = np.zeros(size)
        self.index = 0
        
    def add_chunk(self, chunk):
        """Add a chunk of audio data to the circular buffer"""
        chunk_size = len(chunk)
        if chunk_size >= self.size:
            self.buffer = chunk[-self.size:].copy()
            self.index = 0
        else:
            end_idx = self.index + chunk_size
            if end_idx <= self.size:
                self.buffer[self.index:end_idx] = chunk
                self.index = end_idx % self.size
            else:
                first_part = self.size - self.index
                self.buffer[self.index:] = chunk[:first_part]
                self.buffer[:chunk_size - first_part] = chunk[first_part:]
                self.index = chunk_size - first_part
    
    def get_waveform(self):
        """Get the current waveform data in correct order"""
        if self.index == 0:
            return self.buffer.copy()
        else:
            return np.concatenate([self.buffer[self.index:], self.buffer[:self.index]])

# ---------------- Main SAI Visualization with Wav2Vec2 and WebView Feedback ----------------
class SAIVisualizationWithWav2Vec2:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 debug=True, playback_speed=1.0, loop_audio=True, wav2vec2_model="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
                 tone_analysis_mode="word", use_webview=True):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        self.sai_speed = 1.0
        self.sai_file_index = 0.0

        # Reference text and target phonemes
        self.reference_text = None
        self.reference_pronunciation = None
        self.translated_text = None
        self.target_phonemes = "ɕiɛɕiɛ"

        # Initialize Wav2Vec2 handler and phoneme analyzer
        self.wav2vec2_handler = SimpleWav2Vec2HandlerWithLogging(model_name=wav2vec2_model, target_phonemes=self.target_phonemes)
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)

        self.tone_analyzer = None
        self.tone_analysis_mode = tone_analysis_mode
        self.tone_analysis_results = None

        match self.tone_analysis_mode:
            case "word":
                self.tone_analyzer = ToneAnalyzerWord()
            case "sentence":
                self.tone_analyzer = ToneAnalyzerSentence()

        self.wav2vec2_handler.register_callback(self.tone_analysis_callback)
        
        # WebView feedback display
        self.use_browser_feedback = use_webview  # Keep parameter name for compatibility
        self.feedback_webview = None
        if self.use_browser_feedback:
            self.feedback_webview = BrowserFeedback()
            self.feedback_webview.start()  # This is non-blocking now
            print("Browser feedback initialized")
        
        # Feedback display components
        self.phoneme_feedback_displays = []
        self.feedback_background = None
        
        # Audio processors
        self.processor_realtime = AudioProcessor(fs=sample_rate)
        self.processor_file = AudioProcessor(fs=sample_rate)
        self.n_channels = self.processor_realtime.n_channels

        # SAI parameters
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=self.sai_width,
            future_lags=self.sai_width - 1,
            num_triggers_per_frame=2,
            trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size,
            channel_smoothing_scale=0.5
        )
        
        # SAI processors
        self.sai_realtime = SAIProcessor(self.sai_params)
        self.sai_file = SAIProcessor(self.sai_params)

        # Visualization
        self.vis_realtime = VisualizationHandler(sample_rate, self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, self.sai_params)

        # Waveform buffers
        self.waveform_realtime = WaveformBuffer(size=int(sample_rate * 0.5))
        self.waveform_file = WaveformBuffer(size=int(sample_rate * 0.5))

        # Audio setup
        self.audio_queue = queue.Queue(maxsize=50)
        
        # File processing
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.current_position = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0
        
        # Audio playback
        self.audio_playback_enabled = True
        self.audio_output_stream = None
        self.playback_position = 0.0
        
        if audio_file_path and os.path.exists(audio_file_path):
            self._load_audio_file()
        
        # PyAudio
        self.p = None
        self.stream = None
        self.running = False
        
        self.similarity_display = None
        self.similarity_rect = None
        
        self._setup_dual_visualization()

    def create_webview_feedback_data(self, tone_results, phoneme_results):
        """Convert analysis results to format for HTML interface"""
        
        def split_phonemes_by_syllable(phonemes, num_syllables):
            if not phonemes or num_syllables == 0:
                return []
            phonemes_per_syllable = max(1, len(phonemes) // num_syllables)
            syllables = []
            for i in range(num_syllables):
                start = i * phonemes_per_syllable
                end = start + phonemes_per_syllable
                if start < len(phonemes):
                    syllables.append(phonemes[start:end])
            return syllables
        
        # BUILD TARGET DATA FROM CURRENT PRACTICE ITEM
        target_data = []
        
        if hasattr(self, 'current_practice_item') and self.current_practice_item:
            item = self.current_practice_item
            
            if 'character' in item:  # Word item
                characters = list(item['character'])  # Split: "谢谢" → ['谢', '谢']
                num_chars = len(characters)
                
                # Split phonemes: "ɕiɛɕiɛ" → ['ɕiɛ', 'ɕiɛ']
                phoneme_list = list(item['phonemes'])
                phoneme_groups = split_phonemes_by_syllable(phoneme_list, num_chars)
                
                # Handle tones
                tones = item['tone']
                if isinstance(tones, int):
                    tones = [tones] * num_chars
                
                # Split pinyin by syllables for multi-char words
                pinyin_full = item['pinyin']
                # For 谢谢: "xièxie" should show as "xiè" for each character
                # Simple approach: repeat the pinyin or split by uppercase
                import re
                pinyin_syllables = re.findall(r'[A-Za-z]+', pinyin_full)
                if len(pinyin_syllables) != num_chars:
                    pinyin_syllables = [pinyin_full] * num_chars
                
                for idx, char in enumerate(characters):
                    phonemes = phoneme_groups[idx] if idx < len(phoneme_groups) else []
                    tone = tones[idx] if idx < len(tones) else 1
                    pinyin = pinyin_syllables[idx] if idx < len(pinyin_syllables) else pinyin_full
                    
                    target_data.append({
                        'character': char,
                        'pinyin': pinyin,
                        'tone': tone,
                        'phonemes': phonemes,
                        'phoneme_scores': ['perfect'] * len(phonemes)
                    })
            
            else:  # Sentence item
                # Use tone results for sentences
                if tone_results and len(tone_results) > 0:
                    phoneme_list = list(item['phonemes'])
                    phoneme_groups = split_phonemes_by_syllable(phoneme_list, len(tone_results))
                    
                    for idx, tone_info in enumerate(tone_results):
                        phonemes = phoneme_groups[idx] if idx < len(phoneme_groups) else []
                        target_data.append({
                            'character': tone_info.get('character', '?'),
                            'pinyin': tone_info.get('pinyin', '?'),
                            'tone': tone_info.get('tone', 1),
                            'phonemes': phonemes,
                            'phoneme_scores': ['perfect'] * len(phonemes)
                        })
        
        # BUILD USER DATA (same as before)
        yours_data = []
        if phoneme_results and isinstance(phoneme_results, dict):
            analysis = phoneme_results.get('analysis_results', [])
            
            if analysis and len(analysis) > 0:
                user_phonemes = [r.get('detected', '.') or '.' for r in analysis]
                user_syllables = split_phonemes_by_syllable(user_phonemes, len(target_data))
                
                for idx, target_info in enumerate(target_data):
                    phonemes = user_syllables[idx] if idx < len(user_syllables) else []
                    tone_info = tone_results[idx] if idx < len(tone_results) else {}
                    
                    phonemes_per_syllable = len(phonemes)
                    start_idx = idx * phonemes_per_syllable
                    end_idx = start_idx + phonemes_per_syllable
                    syllable_analysis = analysis[start_idx:end_idx] if start_idx < len(analysis) else []
                    
                    phoneme_scores = []
                    for result in syllable_analysis:
                        status = result.get('status', 'wrong')
                        similarity = result.get('similarity', 0)
                        
                        if status == 'correct' and similarity >= 0.95:
                            phoneme_scores.append('perfect')
                        elif status == 'correct':
                            phoneme_scores.append('good')
                        elif status == 'close':
                            phoneme_scores.append('close')
                        elif status == 'missing':
                            phoneme_scores.append('missing')
                        else:
                            phoneme_scores.append('wrong')
                    
                    while len(phoneme_scores) < len(phonemes):
                        phoneme_scores.append('wrong')
                    
                    yours_data.append({
                        'character': tone_info.get('character', '?'),
                        'pinyin': tone_info.get('pinyin', '?'),
                        'tone': tone_info.get('predicted_tone', None),
                        'confidence': tone_info.get('confidence', 0),
                        'phonemes': phonemes,
                        'phoneme_scores': phoneme_scores[:len(phonemes)]
                    })
        
        return {
            'overall_score': phoneme_results.get('overall_score', 0) if isinstance(phoneme_results, dict) else 0,
            'target': target_data,
            'yours': yours_data
        }

    def decrease_sai_speed(self, event=None):
        self.sai_speed = max(0.1, self.sai_speed - 0.25)
        self.update_sai_speed_display()
        print(f"SAI speed: {self.sai_speed:.1f}x")

    def increase_sai_speed(self, event=None):
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
        self.update_sai_speed_display()
        print(f"SAI speed: {self.sai_speed:.1f}x")

    def update_sai_speed_display(self):
        if hasattr(self, 'sai_speed_display'):
            self.sai_speed_display.set_text(f'SAI Speed: {self.sai_speed:.1f}x')

    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
        self.update_audio_speed_display()
        print(f"Audio speed: {self.playback_speed:.1f}x")

    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)
        self.update_audio_speed_display()
        print(f"Audio speed: {self.playback_speed:.1f}x")

    def update_audio_speed_display(self):
        if hasattr(self, 'audio_speed_display'):
            self.audio_speed_display.set_text(f'Audio Speed: {self.playback_speed:.1f}x')

    def on_key_press(self, event):
        if event.key == 'up' or event.key == '+':
            self.increase_sai_speed()
        elif event.key == 'down' or event.key == '-':
            self.decrease_sai_speed()
        elif event.key == 'right':
            self.increase_audio_speed()
        elif event.key == 'left':
            self.decrease_audio_speed()
        elif event.key == 'r':
            self.sai_speed = 1.0
            self.playback_speed = 1.0
            self.update_sai_speed_display()
            self.update_audio_speed_display()
            print("Both speeds reset to 1.0x")
        elif event.key == 'c':
            self.clear_phoneme_feedback()

    def clear_phoneme_feedback(self):
        for display in self.phoneme_feedback_displays:
            if display is not None:
                display.remove()
        self.phoneme_feedback_displays.clear()
        
        if self.feedback_background is not None:
            self.feedback_background.remove()
            self.feedback_background = None

    def create_phoneme_feedback_display(self, analysis_results, overall_score):
        """Create detailed per-phoneme feedback display"""
        try:
            self.clear_phoneme_feedback()
            
            if not analysis_results:
                return
            
            if not hasattr(analysis_results, '__iter__'):
                print("Analysis results is not iterable")
                return
            
            def get_phoneme_color(result):
                status = result['status']
                similarity = result['similarity']
                
                if status == 'missing':
                    return '#666666'
                elif status == 'extra':
                    return '#FF00FF'
                elif status == 'correct':
                    if similarity >= 0.95:
                        return '#00FF00'
                    else:
                        return '#44FF44'
                elif status == 'close':
                    if similarity >= 0.7:
                        return '#FFAA00'
                    elif similarity >= 0.6:
                        return '#FFCC44'
                    else:
                        return '#FFFF00'
                else:
                    if similarity >= 0.3:
                        return '#FF6600'
                    elif similarity >= 0.1:
                        return '#FF3300'
                    else:
                        return '#CC0000'
            
            def get_overall_score_color(score):
                if score >= 0.9:
                    return '#00FF00'
                elif score >= 0.8:
                    return '#44FF44'
                elif score >= 0.7:
                    return '#88FF00'
                elif score >= 0.6:
                    return '#FFFF00'
                elif score >= 0.5:
                    return '#FFAA00'
                elif score >= 0.3:
                    return '#FF6600'
                else:
                    return '#FF0000'
            
            score_color = get_overall_score_color(overall_score)
            overall_display = self.ax_realtime.text(
                0.5, 0.7,
                f"Overall Score: {overall_score*100:.0f}%",
                transform=self.ax_realtime.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16, fontweight='bold',
                color=score_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white')
            )
            self.phoneme_feedback_displays.append(overall_display)
            
            target_text = "TARGET: " + "".join([r['target'] or '∅' for r in analysis_results])
            target_display = self.ax_realtime.text(
                0.5, 0.62,
                target_text,
                transform=self.ax_realtime.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, fontweight='bold',
                color='cyan',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
            self.phoneme_feedback_displays.append(target_display)
            
            detected_text = "YOURS:  "
            detected_display = self.ax_realtime.text(
                0.1, 0.54,
                detected_text,
                transform=self.ax_realtime.transAxes,
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=14, fontweight='bold',
                color='white'
            )
            self.phoneme_feedback_displays.append(detected_display)
            
            start_x = 0.25
            x_step = 0.08
            
            for i, result in enumerate(analysis_results):
                x_pos = start_x + (i * x_step)
                
                phoneme_char = result['detected'] or '∅'
                color = get_phoneme_color(result)
                
                phoneme_display = self.ax_realtime.text(
                    x_pos, 0.54,
                    phoneme_char,
                    transform=self.ax_realtime.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3)
                )
                self.phoneme_feedback_displays.append(phoneme_display)
                
                if result['similarity'] > 0 and result['similarity'] < 0.8:
                    similarity_text = f"{result['similarity']*100:.0f}%"
                    similarity_display = self.ax_realtime.text(
                        x_pos, 0.50,
                        similarity_text,
                        transform=self.ax_realtime.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color=color,
                        alpha=0.7
                    )
                    self.phoneme_feedback_displays.append(similarity_display)
            
            legend_y = 0.42
            legend_items = [
                ("Perfect", '#00FF00'),
                ("Good", '#44FF44'),
                ("Close", '#FFAA00'),
                ("Poor", '#FF6600'),
                ("Wrong", '#FF0000'),
                ("Missing", '#666666')
            ]
            
            legend_start_x = 0.1
            x_spacing = 0.12
            
            for i, (label, color) in enumerate(legend_items):
                legend_display = self.ax_realtime.text(
                    legend_start_x + i * x_spacing, legend_y,
                    f"● {label}",
                    transform=self.ax_realtime.transAxes,
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=9, fontweight='bold',
                    color=color
                )
                self.phoneme_feedback_displays.append(legend_display)
            
        except Exception as e:
            print(f"Error creating phoneme feedback display: {e}")
            import traceback
            traceback.print_exc()

    def tone_analysis_callback(self, audio):
        """Callback to run tone analysis on audio"""
        if self.tone_analyzer is not None:
            try:
                self.tone_analysis_results = self.tone_analyzer.predict_tone(audio)
                print(f"Tone analysis results: {self.tone_analysis_results}")
                
                # ADD THIS: Update browser feedback
                if (self.use_browser_feedback and self.feedback_webview and 
                    self.tone_analysis_results and 
                    self.wav2vec2_handler.analysis_results):
                    
                    phoneme_results = {
                        'overall_score': self.wav2vec2_handler.overall_score,
                        'analysis_results': self.wav2vec2_handler.analysis_results
                    }
                    
                    try:
                        feedback_data = self.create_webview_feedback_data(
                            self.tone_analysis_results,
                            phoneme_results
                        )
                        self.feedback_webview.update_feedback(feedback_data)
                        print(f"Browser feedback updated - Score: {self.wav2vec2_handler.overall_score:.0%}")
                    except Exception as e:
                        print(f"Error updating browser: {e}")
                        
            except Exception as e:
                print(f"Tone analysis error: {e}")
                self.tone_analysis_results = None

    def _load_audio_file(self):
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, original_sr = librosa.load(self.audio_file_path, sr=None)
        
        if original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
        
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        
        if self.audio_playback_enabled:
            self._setup_audio_playback()

    def set_reference_text(self, phonemes, pronunciation, translation):
        self.reference_text = phonemes.strip()
        self.reference_pronunciation = pronunciation
        self.translated_text = translation.strip()
        print(f"Target word updated: {pronunciation} - {translation}")
        print(f"Target phonemes: {phonemes}")
        
        # ADD THIS: Update the display text immediately
        if hasattr(self, 'transcription_file'):
            reference_display = f"Target word: {self.reference_pronunciation}"
            if self.translated_text and self.translated_text != 'audio file':
                reference_display += f" - {self.translated_text}"
            self.transcription_file.set_text(reference_display)

    def _setup_audio_playback(self):
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback
            )
            print("Audio playback stream created")
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _audio_playback_callback(self, outdata, frames, time, status):
        try:
            if self.audio_data is not None:
                start_pos = int(self.playback_position)
                end_pos = min(start_pos + frames, self.total_samples)

                speed_factor = self.playback_speed
                chunk_indices = np.arange(frames) * speed_factor
                chunk_indices = chunk_indices.astype(int) + start_pos
                chunk_indices = np.clip(chunk_indices, 0, self.total_samples - 1)
                chunk = self.audio_data[chunk_indices]

                outdata[:, 0] = chunk

                self.playback_position += int(frames * speed_factor)
                if self.playback_position >= self.total_samples:
                    if self.loop_audio:
                        self.playback_position = 0
                    else:
                        outdata.fill(0)
            else:
                outdata.fill(0)
        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata.fill(0)

    def get_next_file_chunk(self):
        if self.audio_data is None:
            return None, -1
        
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
            else:
                return None, -1
        
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        chunk = self.audio_data[self.current_position:end_position]
        
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        
        chunk_index = self.current_position
        self.current_position = end_position
        
        return chunk.astype(np.float32), chunk_index

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for SAI visualization only"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.waveform_realtime.add_chunk(audio_float)
            
            try:
                self.audio_queue.put_nowait(audio_float)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_float)
                except queue.Empty:
                    pass
        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def process_realtime_audio(self):
        """Process real-time audio for SAI visualization only"""
        print("Real-time SAI processing started")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                nap_output = self.processor_realtime.process_chunk(audio_chunk)
                sai_output = self.sai_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)

                self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time processing error: {e}")
                continue

    def toggle_phoneme_recognition(self, event=None):
        if not self.wav2vec2_handler.is_recording:
            if self.wav2vec2_handler.start_recording():
                self.btn_transcribe.label.set_text('Stop Recognition')
                print("Started recording for phoneme recognition")
                self.clear_phoneme_feedback()
        else:
            self.wav2vec2_handler.stop_recording()
            self.btn_transcribe.label.set_text('check your pronunciation score')
            print("Stopped recording")

    def _setup_dual_visualization(self):
        print("DEBUG: Starting _setup_dual_visualization")
        self.fig = plt.figure(figsize=(16, 14))
        gs = self.fig.add_gridspec(12, 2, height_ratios=[1]*8 + [0.5, 0.5, 0.3, 0.3])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:8, 0])
        self.ax_file = self.fig.add_subplot(gs[0:8, 1])
        
        self.ax_waveform_realtime = self.fig.add_subplot(gs[8, 0])
        self.ax_waveform_file = self.fig.add_subplot(gs[8, 1])
        
        print("DEBUG: Created axes")
        
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_realtime.axis('off')
        
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_file.axis('off')
        
        waveform_length = self.waveform_realtime.size
        time_axis = np.linspace(0, waveform_length / self.sample_rate, waveform_length)
        
        self.line_waveform_realtime, = self.ax_waveform_realtime.plot(
            time_axis, np.zeros(waveform_length), 'lime', linewidth=1
        )
        self.ax_waveform_realtime.set_xlim(0, waveform_length / self.sample_rate)
        self.ax_waveform_realtime.set_ylim(-1, 1)
        self.ax_waveform_realtime.tick_params(colors='white', labelsize=8)
        self.ax_waveform_realtime.set_facecolor('black')
        
        self.line_waveform_file, = self.ax_waveform_file.plot(
            time_axis, np.zeros(waveform_length), 'cyan', linewidth=1
        )
        self.ax_waveform_file.set_xlim(0, waveform_length / self.sample_rate)
        self.ax_waveform_file.set_ylim(-1, 1)
        self.ax_waveform_file.tick_params(colors='white', labelsize=8)
        self.ax_waveform_file.set_facecolor('black')
        
        print("DEBUG: Created images and waveforms")
        
        print("DEBUG: Creating text overlays...")
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, 'Live SAI', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=12, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_realtime")
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=12, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_file")
        
        self.transcription_status = self.ax_realtime.text(
            0.02, 0.12, 'Wav2Vec2 Phonemes: Click to start', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='orange', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_status")
        
        self.ax_controls = self.fig.add_subplot(gs[10:12, :])
        self.ax_controls.axis('off')
        
        button_width = 0.12
        button_height = 0.04
        bottom_margin = 0.02
        top_row_y = bottom_margin + button_height + 0.01
        bottom_row_y = bottom_margin
        
        self.ax_playback = plt.axes([0.35, top_row_y, button_width, button_height])
        self.btn_playback = Button(self.ax_playback, 'Play Audio', 
                                color='lightgreen', hovercolor='green')
        self.btn_playback.on_clicked(self.toggle_playback)
        
        self.ax_transcribe = plt.axes([0.50, top_row_y, button_width + 0.03, button_height])
        self.btn_transcribe = Button(self.ax_transcribe, 'Check Pronunciation Score', 
                                    color='lightblue', hovercolor='blue')
        self.btn_transcribe.on_clicked(self.toggle_phoneme_recognition)

        try:
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, hspace=0.3, wspace=0.3)
        except Exception as e:
            print(f"Layout adjustment warning: {e}")
        
        self.fig.patch.set_facecolor('black')
        print("DEBUG: Finished _setup_dual_visualization")

    def toggle_playback(self, event=None):
        if self.audio_playback_enabled and self.audio_output_stream:
            if self.audio_output_stream.active:
                self.audio_output_stream.stop()
                self.btn_playback.label.set_text('Play Audio')
                print("Reference playback stopped")
            else:
                self.audio_output_stream.start()
                self.btn_playback.label.set_text('Stop Audio')
                print("Reference playback started")
        else:
            print("Reference playback not available")

    def update_visualization(self, frame):
        try:
            if self.vis_realtime.img is not None and self.vis_realtime.img.size > 0:
                current_max_rt = np.max(self.vis_realtime.img)
            else:
                current_max_rt = 1

            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))

            waveform_data = self.waveform_realtime.get_waveform()
            self.line_waveform_realtime.set_ydata(waveform_data)

            # try:
            #     detected = getattr(self.wav2vec2_handler, 'result', None)

            #     if detected and detected != "no_audio":
            #         analysis_results = getattr(self.wav2vec2_handler, 'analysis_results', [])
            #         overall_score = getattr(self.wav2vec2_handler, 'overall_score', 0.0)
                    
            #         if analysis_results and len(analysis_results) > 0:
            #             self.create_phoneme_feedback_display(analysis_results, overall_score)

            # except Exception as e:
            #     print(f"DEBUG: Error updating phoneme feedback: {e}")

            # Update file SAI visualization
            if self.audio_data is not None:
                chunk, chunk_index = self.get_next_file_chunk()
                if chunk is not None and chunk_index >= 0:
                    try:
                        self.waveform_file.add_chunk(chunk)
                        
                        nap_output = self.processor_file.process_chunk(chunk)
                        sai_output = self.sai_file.RunSegment(nap_output)
                        self.vis_file.get_vowel_embedding(nap_output)
                        self.vis_file.run_frame(sai_output)

                        self.sai_file_index += self.sai_speed
                        
                        if self.sai_file_index >= 1.0:
                            steps = int(self.sai_file_index)
                            self.sai_file_index -= steps
                            
                            for _ in range(steps):
                                if self.vis_file.img.shape[1] > 1:
                                    self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                                self.vis_file.draw_column(self.vis_file.img[:, -1])

                    except Exception as e:
                        print(f"Error processing file chunk: {e}")

            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))

            file_waveform_data = self.waveform_file.get_waveform()
            self.line_waveform_file.set_ydata(file_waveform_data)

            reference_display = ''
            if self.reference_pronunciation:
                reference_display = f"Target word: {self.reference_pronunciation}"
            if self.translated_text and self.translated_text != 'audio file':
                reference_display += f" - {self.translated_text}"
            self.transcription_file.set_text(reference_display)

            status_text, status_color = self.wav2vec2_handler.get_current_status()
            if status_text is not None:
                self.transcription_status.set_text(status_text)
                self.transcription_status.set_color(status_color)

        except Exception as e:
            print(f"Visualization update error: {e}")

        try:
            elements_to_return = []
            
            main_elements = [
                ('im_realtime', self.im_realtime),
                ('im_file', self.im_file),
                ('line_waveform_realtime', self.line_waveform_realtime),
                ('line_waveform_file', self.line_waveform_file),
                ('transcription_realtime', self.transcription_realtime),
                ('transcription_file', self.transcription_file),
                ('transcription_status', self.transcription_status)
            ]
            
            for name, elem in main_elements:
                if elem is not None and hasattr(elem, 'axes'):
                    elements_to_return.append(elem)
                elif elem is not None:
                    try:
                        if hasattr(elem, 'set_data') or hasattr(elem, 'set_text') or hasattr(elem, 'set_ydata'):
                            elements_to_return.append(elem)
                    except:
                        pass
            
            # if hasattr(self, 'phoneme_feedback_displays') and self.phoneme_feedback_displays:
            #     valid_displays = []
            #     for display in self.phoneme_feedback_displays:
            #         if display is not None:
            #             try:
            #                 if hasattr(display, 'set_text') and hasattr(display, 'get_text'):
            #                     valid_displays.append(display)
            #             except:
            #                 pass
            #     elements_to_return.extend(valid_displays)
            
            # if (hasattr(self, 'feedback_background') and 
            #     self.feedback_background is not None and 
            #     hasattr(self.feedback_background, 'set_visible')):
            #     elements_to_return.append(self.feedback_background)

            return elements_to_return
            
        except Exception as return_error:
            print(f"Error preparing return elements: {return_error}")
            safe_elements = []
            if hasattr(self, 'im_realtime') and self.im_realtime is not None:
                safe_elements.append(self.im_realtime)
            if hasattr(self, 'im_file') and self.im_file is not None:
                safe_elements.append(self.im_file)
            return safe_elements

    def start(self):
        """Start the visualization"""
        print(f"Target phonemes: {self.target_phonemes}")
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return

        self.running = True
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        # Start streams
        if self.stream:
            self.stream.start_stream()
            print("Audio input stream started")
        
        if self.audio_playback_enabled and self.audio_output_stream:
            self.audio_output_stream.start()
            print("Reference audio playback started")
        
        # No webview threading - just run matplotlib normally
        animation_interval = max(10, int((self.chunk_size / self.sample_rate) * 1000))
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        
        plt.show()  # This runs on main thread - NO THREADING!

    def cleanup(self):
        self.running = False
        
        if self.wav2vec2_handler.is_recording:
            self.wav2vec2_handler.stop_recording()
        
        self.clear_phoneme_feedback()
        
        if self.audio_output_stream:
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except:
                pass
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except:
            pass

        if hasattr(self.wav2vec2_handler, 'end_session'):
            self.wav2vec2_handler.end_session()

                # Add this to prevent the Tkinter error:
        try:
            plt.close('all')
        except:
            pass

    def stop(self):
        self.cleanup()
        plt.close('all')

# ---------------- Voice Selector ----------------
class VoiceSelector:
    """Simple voice selector for switching between pre-recorded male/female audio files"""
    
    def __init__(self, male_audio_path, female_audio_path):
        self.male_audio_path = male_audio_path
        self.female_audio_path = female_audio_path
        self.current_voice = "female"
        
        self.male_available = os.path.exists(male_audio_path) if male_audio_path else False
        self.female_available = os.path.exists(female_audio_path) if female_audio_path else False
        
        print(f"Voice files - Male: {self.male_available}, Female: {self.female_available}")
    
    def get_current_audio_path(self):
        if self.current_voice == "male" and self.male_available:
            return self.male_audio_path
        elif self.current_voice == "female" and self.female_available:
            return self.female_audio_path
        else:
            return None
    
    def switch_to_male(self):
        if self.male_available:
            self.current_voice = "male"
            return self.male_audio_path
        return None
    
    def switch_to_female(self):
        if self.female_available:
            self.current_voice = "female"
            return self.female_audio_path
        return None

# ---------------- Voice Selection Enhanced Class ----------------
class SAIVisualizationWithVoiceSelection(SAIVisualizationWithWav2Vec2):
    """Enhanced version with voice file switching"""
    
    def __init__(self, male_audio_path=None, female_audio_path=None, audio_manager=None, *args, **kwargs):
        # Store audio_manager BEFORE calling super().__init__
        self.audio_manager = audio_manager  # <-- This line should be FIRST, BEFORE super()
        
        self.voice_selector = VoiceSelector(male_audio_path, female_audio_path)
        
        initial_audio = self.voice_selector.get_current_audio_path()
        if initial_audio:
            kwargs['audio_file_path'] = initial_audio
        
        # Call parent init WITHOUT audio_manager (parent doesn't accept it)
        # DON'T pass audio_manager to super() - it's already been extracted
        super().__init__(*args, **kwargs)
    
    def _setup_dual_visualization(self):
        print("DEBUG: Starting _setup_dual_visualization with voice selection")
        super()._setup_dual_visualization()
        self._add_voice_selection_buttons()
    
    def _add_voice_selection_buttons(self):
        voice_button_width = 0.06
        voice_button_height = 0.04
        voice_start_x = 0.08
        voice_y = 0.02
        
        male_color = 'lightcyan' if self.voice_selector.current_voice == 'male' else 'lightgray'
        male_enabled = self.voice_selector.male_available
        
        self.ax_voice_male = plt.axes([voice_start_x, voice_y, voice_button_width, voice_button_height])
        self.btn_voice_male = Button(self.ax_voice_male, 'Male', 
                                    color=male_color if male_enabled else 'darkgray',
                                    hovercolor='cyan' if male_enabled else 'darkgray')
        if male_enabled:
            self.btn_voice_male.on_clicked(self.select_male_voice)
        
        female_color = 'lightpink' if self.voice_selector.current_voice == 'female' else 'lightgray'
        female_enabled = self.voice_selector.female_available
        
        self.ax_voice_female = plt.axes([voice_start_x + voice_button_width + 0.01, voice_y, voice_button_width, voice_button_height])
        self.btn_voice_female = Button(self.ax_voice_female, 'Female',
                                    color=female_color if female_enabled else 'darkgray', 
                                    hovercolor='pink' if female_enabled else 'darkgray')
        if female_enabled:
            self.btn_voice_female.on_clicked(self.select_female_voice)
        
        voice_status = f"Voice: {self.voice_selector.current_voice.title()}"
        self.voice_status_display = self.ax_controls.text(
            0.91, 0.05, voice_status,
            transform=self.ax_controls.transAxes,
            fontsize=10, color='lightgreen', weight='bold', horizontalalignment='center'
        )
    
    def select_male_voice(self, event=None):
        new_audio_path = self.voice_selector.switch_to_male()
        if new_audio_path:
            self._switch_audio_file(new_audio_path)
            self._update_voice_displays()
            print(f"Switched to male voice: {os.path.basename(new_audio_path)}")
    
    def select_female_voice(self, event=None):
        new_audio_path = self.voice_selector.switch_to_female()
        if new_audio_path:
            self._switch_audio_file(new_audio_path)
            self._update_voice_displays()
            print(f"Switched to female voice: {os.path.basename(new_audio_path)}")
    
    def _switch_audio_file(self, new_audio_path):
        try:
            if self.audio_output_stream and self.audio_output_stream.active:
                self.audio_output_stream.stop()
            
            self.audio_file_path = new_audio_path
            self._load_audio_file()
            
            self.current_position = 0
            self.playback_position = 0.0
            
        except Exception as e:
            print(f"Error switching audio file: {e}")
    
    def _update_voice_displays(self):
        current_voice = self.voice_selector.current_voice
        
        if hasattr(self, 'btn_voice_male'):
            male_color = 'lightcyan' if current_voice == 'male' else 'lightgray'
            self.btn_voice_male.color = male_color
            self.btn_voice_male.ax.set_facecolor(male_color)
        
        if hasattr(self, 'btn_voice_female'):
            female_color = 'lightpink' if current_voice == 'female' else 'lightgray'
            self.btn_voice_female.color = female_color
            self.btn_voice_female.ax.set_facecolor(female_color)
        
        if hasattr(self, 'voice_status_display'):
            voice_status = f"Voice: {current_voice.title()}"
            self.voice_status_display.set_text(voice_status)

# ---------------- Pronunciation Logger ----------------
class PronunciationLogger:
    """Logger for pronunciation learning scores with timestamps"""
    
    def __init__(self, log_directory="pronunciation_logs", log_filename=None):
        # Get absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_directory = os.path.join(script_dir, log_directory)
        
        # Create directory if it doesn't exist - THIS IS THE KEY FIX
        os.makedirs(self.log_directory, exist_ok=True)
        
        if log_filename is None:
            today = datetime.now().strftime("%Y-%m-%d")
            log_filename = f"pronunciation_scores_{today}.txt"
        
        self.log_file_path = os.path.join(self.log_directory, log_filename)
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write("=== Pronunciation Learning Log ===\n")
                f.write(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Format: [Timestamp] Target -> Detected | Score: X.XX | Details\n")
                f.write("-" * 60 + "\n\n")
    
    def log_score(self, target_phonemes, detected_phonemes, overall_score, analysis_results=None):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {target_phonemes} -> {detected_phonemes} | Score: {overall_score:.2f}\n"
            
            if analysis_results and len(analysis_results) > 0:
                log_entry += "  Details: "
                phoneme_details = []
                for result in analysis_results:
                    target = result.get('target', '?')
                    detected = result.get('detected', '?')
                    status = result.get('status', 'unknown')
                    
                    if target and detected:
                        if status == 'correct':
                            phoneme_details.append(f"{target}✓")
                        elif status == 'close':
                            phoneme_details.append(f"{target}~{detected}")
                        elif status == 'incorrect':
                            phoneme_details.append(f"{target}✗{detected}")
                        elif status == 'missing':
                            phoneme_details.append(f"{target}?")
                        elif status == 'extra':
                            phoneme_details.append(f"+{detected}")
                
                log_entry += " | ".join(phoneme_details) + "\n"
            
            log_entry += "\n"
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print(f"Score logged to: {self.log_file_path}")
            
        except Exception as e:
            print(f"Error logging score: {e}")
    
    def log_session_summary(self, session_scores):
        try:
            if not session_scores:
                return
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_score = sum(session_scores) / len(session_scores)
            max_score = max(session_scores)
            attempts = len(session_scores)
            
            summary = f"\n--- Session Summary [{timestamp}] ---\n"
            summary += f"Attempts: {attempts}\n"
            summary += f"Average Score: {avg_score:.2f}\n"
            summary += f"Best Score: {max_score:.2f}\n"
            summary += f"Progress: {session_scores}\n"
            summary += "-" * 40 + "\n\n"
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"Session summary logged")
            
        except Exception as e:
            print(f"Error logging session summary: {e}")
    
    def get_log_file_path(self):
        return self.log_file_path

# ---------------- Handler with Logging ----------------
class SimpleWav2Vec2HandlerWithLogging(SimpleWav2Vec2Handler):
    """Enhanced Wav2Vec2 handler with pronunciation logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.logger = PronunciationLogger()
        self.session_scores = []
        
        print(f"Pronunciation logging enabled: {self.logger.get_log_file_path()}")
    
    def _process_audio(self):
        super()._process_audio()
        
        if (hasattr(self, 'result') and self.result and 
            self.result not in ["no_audio", "processing_error"] and
            hasattr(self, 'analysis_results') and self.analysis_results and
            hasattr(self, 'overall_score')):
            
            self.logger.log_score(
                target_phonemes=self.target_phonemes,
                detected_phonemes=self.result,
                overall_score=self.overall_score,
                analysis_results=self.analysis_results
            )
            
            self.session_scores.append(self.overall_score)
            
            if len(self.session_scores) % 5 == 0:
                self.logger.log_session_summary(self.session_scores)
            
            # ADD THIS: Trigger browser feedback update
            self._trigger_browser_feedback()

    def _trigger_browser_feedback(self):
        """Trigger browser feedback update after processing"""
        # This will be called from the parent visualization
        pass  # The parent class will handle this
    
    def end_session(self):
        if self.session_scores:
            self.logger.log_session_summary(self.session_scores)
            print(f"Final session summary logged with {len(self.session_scores)} attempts")

class ReferenceAudioManager:
    """Manages organized reference audio files matching mandarin_vocab.json"""
    
    def __init__(self, reference_dir="reference", vocab_db=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.reference_dir = os.path.join(script_dir, reference_dir)
        self.vocab_db = vocab_db or VOCAB_DB
        self.audio_files = {}
        self.sentence_audio_files = {}
        self._scan_audio_files()
        self._scan_sentence_audio_files()
    
    def _scan_audio_files(self):
        """Scan reference directory with men/women subfolders for WORDS"""
        if not os.path.exists(self.reference_dir):
            print(f"Creating reference directory: {self.reference_dir}")
            os.makedirs(self.reference_dir, exist_ok=True)
            return
        
        men_dir = os.path.join(self.reference_dir, 'men')
        women_dir = os.path.join(self.reference_dir, 'women')
        
        for word_char, word_info in self.vocab_db['words'].items():
            word_id = word_info.get('id')
            if not word_id:
                continue
            
            self.audio_files[word_char] = {
                'men': [],
                'women': [],
                'info': word_info
            }
            
            # Scan men folder
            if os.path.exists(men_dir):
                for audio_file in os.listdir(men_dir):
                    if not audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        continue
                    
                    if f'{word_id}_' in audio_file or f'_{word_id}.' in audio_file or audio_file.startswith(f'{word_id}.'):
                        full_path = os.path.join(men_dir, audio_file)
                        self.audio_files[word_char]['men'].append({
                            'path': full_path,
                            'filename': audio_file
                        })
            
            # Scan women folder
            if os.path.exists(women_dir):
                for audio_file in os.listdir(women_dir):
                    if not audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        continue
                    
                    if f'{word_id}_' in audio_file or f'_{word_id}.' in audio_file or audio_file.startswith(f'{word_id}.'):
                        full_path = os.path.join(women_dir, audio_file)
                        self.audio_files[word_char]['women'].append({
                            'path': full_path,
                            'filename': audio_file
                        })
            
            total = len(self.audio_files[word_char]['men']) + len(self.audio_files[word_char]['women'])
            if total > 0:
                print(f"Found {total} audio files for '{word_char}' (ID {word_id}, {word_info['pinyin']})")
    
    def _scan_sentence_audio_files(self):
        """Scan reference directory with men/women subfolders for SENTENCES"""
        men_dir = os.path.join(self.reference_dir, 'men')
        women_dir = os.path.join(self.reference_dir, 'women')
        
        for sentence in self.vocab_db['sentences']:
            sentence_id = sentence.get('id')
            if not sentence_id:
                continue
            
            self.sentence_audio_files[sentence_id] = {
                'men': [],
                'women': [],
                'info': sentence
            }
            
            # Scan men folder
            if os.path.exists(men_dir):
                for audio_file in os.listdir(men_dir):
                    if not audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        continue
                    
                    if f'{sentence_id}_' in audio_file or f'_{sentence_id}.' in audio_file or audio_file.startswith(f'{sentence_id}.'):
                        full_path = os.path.join(men_dir, audio_file)
                        self.sentence_audio_files[sentence_id]['men'].append({
                            'path': full_path,
                            'filename': audio_file
                        })
            
            # Scan women folder
            if os.path.exists(women_dir):
                for audio_file in os.listdir(women_dir):
                    if not audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        continue
                    
                    if f'{sentence_id}_' in audio_file or f'_{sentence_id}.' in audio_file or audio_file.startswith(f'{sentence_id}.'):
                        full_path = os.path.join(women_dir, audio_file)
                        self.sentence_audio_files[sentence_id]['women'].append({
                            'path': full_path,
                            'filename': audio_file
                        })
            
            total = len(self.sentence_audio_files[sentence_id]['men']) + len(self.sentence_audio_files[sentence_id]['women'])
            if total > 0:
                mandarin = sentence['mandarin']
                print(f"Found {total} audio files for sentence ID {sentence_id}: '{mandarin}'")
    
    def get_audio_for_word(self, word, voice_type='women', index=0):
        """Get specific audio file for a word
        
        Args:
            word: Chinese character(s)
            voice_type: 'men' or 'women'
            index: Which audio file if multiple exist (0 = first)
        """
        if word not in self.audio_files:
            return None
        
        files = self.audio_files[word].get(voice_type, [])
        if not files:
            other_type = 'men' if voice_type == 'women' else 'women'
            files = self.audio_files[word].get(other_type, [])
        
        if files and index < len(files):
            return files[index]['path']
        
        return None
    
    def get_audio_for_sentence(self, sentence_id, voice_type='women', index=0):
        """Get specific audio file for a sentence by ID
        
        Args:
            sentence_id: Sentence ID number (e.g., 17, 18, etc.)
            voice_type: 'men' or 'women'
            index: Which audio file if multiple exist (0 = first)
        """
        if sentence_id not in self.sentence_audio_files:
            return None
        
        files = self.sentence_audio_files[sentence_id].get(voice_type, [])
        if not files:
            other_type = 'men' if voice_type == 'women' else 'women'
            files = self.sentence_audio_files[sentence_id].get(other_type, [])
        
        if files and index < len(files):
            return files[index]['path']
        
        return None
    
    def get_word_info_with_audio(self, word):
        """Get complete word info including available audio files"""
        if word not in self.vocab_db['words']:
            return None
        
        info = self.vocab_db['words'][word].copy()
        info['character'] = word
        
        if word in self.audio_files:
            info['has_audio'] = True
            info['men_count'] = len(self.audio_files[word]['men'])
            info['women_count'] = len(self.audio_files[word]['women'])
        else:
            info['has_audio'] = False
            info['men_count'] = 0
            info['women_count'] = 0
        
        return info
    
    def list_words_with_audio(self):
        """List all words that have reference audio"""
        return [word for word, files in self.audio_files.items() 
                if len(files['men']) > 0 or len(files['women']) > 0]
    
    def list_all_vocab_words(self):
        """List all words in vocabulary (whether they have audio or not)"""
        return list(self.vocab_db['words'].keys())
    
    def get_missing_audio_words(self):
        """Find words in vocab that don't have audio files"""
        all_words = set(self.list_all_vocab_words())
        words_with_audio = set(self.list_words_with_audio())
        return list(all_words - words_with_audio)

# ---------------- Main Function ----------------
def main_with_voice_selection():
    """Enhanced main function with voice file selection and webview"""
    parser = argparse.ArgumentParser(description='SAI Visualization + Voice Selection + Phoneme Recognition + WebView Feedback')
    
    parser.add_argument('--male-audio', help='Path to male voice audio file')
    parser.add_argument('--female-audio', help='Path to female voice audio file') 
    parser.add_argument('--reference-dir', default='reference', 
                        help='Directory containing organized reference audio')
    parser.add_argument('--word', help='Specific word to practice (e.g., 书, 谢谢)')
    parser.add_argument('--list-words', action='store_true',
                        help='List all available words with audio')
    parser.add_argument('--list-missing', action='store_true',
                        help='List words without audio files')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size for SAI')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width')
    parser.add_argument('--speed', type=float, default=1.0, help='Reference playback speed')
    parser.add_argument('--no-loop', action='store_true', help='Disable reference looping')
    parser.add_argument('--wav2vec2-model', default='facebook/wav2vec2-xlsr-53-espeak-cv-ft',
                        help='Wav2Vec2 model for phoneme recognition')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--default-voice', choices=['male', 'female'], default='female',
                        help='Default voice to start with')
    parser.add_argument('--tone-mode', choices=['word', 'sentence'], default='sentence',
                        help='Which tone analyzer to use')
    parser.add_argument('--no-webview', action='store_true', help='Disable web-based feedback display')
    
    args = parser.parse_args()
    
    # CREATE AUDIO MANAGER FIRST - This is the fix!
    audio_manager = ReferenceAudioManager(args.reference_dir, VOCAB_DB)
    
    # Handle list commands
    if args.list_words:
        words = audio_manager.list_words_with_audio()
        print(f"\nWords with audio ({len(words)}):")
        for word in words:
            info = audio_manager.get_word_info_with_audio(word)
            print(f"  {word} ({info['pinyin']}) - {info['english']}")
            print(f"    Men: {info['men_count']}, Women: {info['women_count']}")
        return 0
    
    if args.list_missing:
        missing = audio_manager.get_missing_audio_words()
        print(f"\nWords missing audio ({len(missing)}):")
        for word in missing:
            info = VOCAB_DB['words'][word]
            print(f"  {word} ({info['pinyin']}) - {info['english']}")
        return 0
    
    # Check if using organized audio or legacy paths
    available_words = audio_manager.list_words_with_audio()
    
    # Determine audio source
    if args.word and available_words:
        # Using organized reference directory
        if args.word not in available_words:
            print(f"\nError: '{args.word}' not found or has no audio")
            print(f"Available: {', '.join(available_words)}")
            return 1
        
        practice_word = args.word
        word_info = audio_manager.get_word_info_with_audio(practice_word)
        
        print(f"\n=== Practicing: {word_info['character']} ===")
        print(f"Pinyin: {word_info['pinyin']}")
        print(f"English: {word_info['english']}")
        print(f"Phonemes: {word_info['phonemes']}")
        print(f"Tone: {word_info['tone']}")
        print(f"Audio files: {word_info['men_count']} men, {word_info['women_count']} women")
        
        men_audio = audio_manager.get_audio_for_word(practice_word, 'men', 0)
        women_audio = audio_manager.get_audio_for_word(practice_word, 'women', 0)
        
        if not men_audio and not women_audio:
            print("Error: No audio files found for this word")
            return 1
    
    elif available_words and not args.male_audio and not args.female_audio:
        # Auto-select first available word
        practice_word = available_words[0]
        word_info = audio_manager.get_word_info_with_audio(practice_word)
        
        print(f"\n=== Auto-selected: {word_info['character']} ===")
        print(f"Pinyin: {word_info['pinyin']}")
        print(f"English: {word_info['english']}")
        print(f"Use --word to choose a different word")
        
        men_audio = audio_manager.get_audio_for_word(practice_word, 'men', 0)
        women_audio = audio_manager.get_audio_for_word(practice_word, 'women', 0)
    
    else:
        # Legacy mode: using --male-audio and --female-audio arguments
        men_audio = args.male_audio if args.male_audio and os.path.exists(args.male_audio) else None
        women_audio = args.female_audio if args.female_audio and os.path.exists(args.female_audio) else None
        
        if not men_audio and not women_audio:
            print("Error: No valid audio files found.")
            print("Use either:")
            print("  --word 书  (to use organized reference directory)")
            print("  --male-audio path.wav --female-audio path.wav  (for custom files)")
            return 1
        
        word_info = None  # No vocab info in legacy mode
    
    # Create visualization
    try:
        sai_vis = SAIVisualizationWithVoiceSelection(
            male_audio_path=men_audio,
            female_audio_path=women_audio,
            audio_manager=audio_manager,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            debug=args.debug,
            playback_speed=args.speed,
            loop_audio=not args.no_loop,
            wav2vec2_model=args.wav2vec2_model,
            tone_analysis_mode=args.tone_mode,
            use_webview=not args.no_webview
        )
        
        # Set target for practice if using vocab
        if word_info:
            sai_vis.set_reference_text(
                word_info['phonemes'],
                word_info['pinyin'],
                word_info['english']
            )
            sai_vis.target_phonemes = word_info['phonemes']
            sai_vis.wav2vec2_handler.target_phonemes = word_info['phonemes']
            sai_vis.wav2vec2_handler.phoneme_analyzer = PhonemeAnalyzer(word_info['phonemes'])
            sai_vis.current_practice_item = word_info
        
        print(f"Starting with {sai_vis.voice_selector.current_voice} voice")
        sai_vis.start()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error starting visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'sai_vis' in locals():
            sai_vis.stop()
        print("Visualization stopped cleanly")
    
    return 0

SAIVisualizationWithVoiceSelection = enhance_visualization_with_practice(
    SAIVisualizationWithVoiceSelection
)
# ---------------- Main Entry Point ----------------
if __name__ == "__main__":
    import sys
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_male = os.path.join(script_dir, "reference", "mandarin_thankyou.mp3")
    default_female = os.path.join(script_dir, "reference", "mandarin_mu.mp3")
    
    if '--male-audio' not in sys.argv and os.path.exists(default_male):
        sys.argv.extend(['--male-audio', default_male])
    
    if '--female-audio' not in sys.argv and os.path.exists(default_female):
        sys.argv.extend(['--female-audio', default_female])
    
    # ADD THIS LINE TO FORCE WEBVIEW ON:
    if '--no-webview' not in sys.argv:
        print("DEBUG: Webview should be enabled")
    
    sys.exit(main_with_voice_selection() or 0)
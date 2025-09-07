from flask import Flask, render_template, jsonify, request, make_response
import subprocess
import os
import threading
from pathlib import Path
import json
import sys

app = Flask(__name__)

# Flask configuration
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configuration - Updated to match your working directory structure
SCRIPT_BASE_PATH = Path(r"C:\Users\maruk\Downloads\Python310\carfac-SAI\python\src")
PYTHON_EXECUTABLE = "python"

# Store running processes (optional - for process management)
running_processes = {}

# Add CSP headers to allow JavaScript execution
@app.after_request
def after_request(response):
    # Allow unsafe inline and eval for development
    response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'"
    # Disable caching for development
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    # Allow CORS for local development
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/')
def index():
    return render_template('pitchogram.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        data = request.get_json()
        script_name = data.get('script')
        display_text = data.get('display_text', script_name)
        
        print(f"Attempting to run: {script_name}")
        print(f"Display text: {display_text}")
        
        if not script_name:
            return jsonify({'success': False, 'message': 'No script specified'}), 400
        
        # Updated path logic to match your working command structure
        if script_name == 'mandarin_mu.py':
            # Use the same structure as your working command: python carfac\mandarin_mu.py
            script_relative_path = f"carfac\\{script_name}"
            script_full_path = SCRIPT_BASE_PATH / 'carfac' / script_name
        else:
            # For other scripts, assume they're in the carfac directory too
            script_relative_path = f"carfac\\{script_name}"
            script_full_path = SCRIPT_BASE_PATH / 'carfac' / script_name
        
        print(f"Script relative path: {script_relative_path}")
        print(f"Script full path: {script_full_path}")
        print(f"File exists: {script_full_path.exists()}")
        print(f"Working directory will be: {SCRIPT_BASE_PATH}")
        
        if not script_full_path.exists():
            # List available files for debugging
            carfac_dir = SCRIPT_BASE_PATH / 'carfac'
            if carfac_dir.exists():
                available_files = [f.name for f in carfac_dir.glob("*.py") if f.is_file()]
                print(f"Available files in carfac directory: {available_files}")
            else:
                available_files = []
                print("Carfac directory does not exist")
            
            return jsonify({
                'success': False, 
                'message': f'Script {script_name} not found at {script_full_path}. Available files in carfac/: {available_files}'
            }), 404
        
        # Run the script in a separate process - mimic your working command exactly
        try:
            if os.name == 'nt':  # Windows
                # Use the same command structure that works: python carfac\mandarin_mu.py
                # Run from the src directory, just like your manual command
                process = subprocess.Popen(
                    [PYTHON_EXECUTABLE, script_relative_path],
                    cwd=str(SCRIPT_BASE_PATH),  # Run from src directory
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    encoding='utf-8'
                )
            else:  # Linux/Mac
                process = subprocess.Popen(
                    [PYTHON_EXECUTABLE, script_relative_path],
                    cwd=str(SCRIPT_BASE_PATH)
                )
            
            # Store process for potential management
            running_processes[script_name] = process
            
            print(f"Successfully started process with PID: {process.pid}")
            print(f"Command executed: {PYTHON_EXECUTABLE} {script_relative_path}")
            print(f"Working directory: {SCRIPT_BASE_PATH}")
            
            return jsonify({
                'success': True,
                'message': f'Successfully started {display_text} pronunciation trainer',
                'script': script_name,
                'pid': process.pid,
                'command': f'{PYTHON_EXECUTABLE} {script_relative_path}',
                'cwd': str(SCRIPT_BASE_PATH)
            })
            
        except FileNotFoundError:
            return jsonify({
                'success': False,
                'message': 'Python interpreter not found. Make sure Python is installed and in PATH.'
            }), 500
            
        except Exception as e:
            print(f"Error starting script: {e}")
            return jsonify({
                'success': False,
                'message': f'Error starting script: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/check-script/<script_name>')
def check_script(script_name):
    """Check if a script exists"""
    script_path = SCRIPT_BASE_PATH / 'carfac' / script_name
    exists = script_path.exists()
    
    return jsonify({
        'exists': exists,
        'path': str(script_path),
        'script': script_name,
        'base_path': str(SCRIPT_BASE_PATH)
    })

@app.route('/list-scripts')
def list_scripts():
    """List available Python scripts"""
    try:
        scripts = []
        carfac_dir = SCRIPT_BASE_PATH / 'carfac'
        
        if carfac_dir.exists():
            for file in carfac_dir.glob("*.py"):
                if file.is_file():
                    scripts.append({
                        'name': file.name,
                        'path': str(file),
                        'size': file.stat().st_size
                    })
        
        return jsonify({
            'success': True,
            'scripts': scripts,
            'base_path': str(SCRIPT_BASE_PATH),
            'carfac_path': str(carfac_dir),
            'carfac_exists': carfac_dir.exists()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/test-js')
def test_js():
    """Test JavaScript execution"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>JavaScript Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            button { padding: 10px 20px; font-size: 16px; margin: 10px; }
            #result { margin-top: 20px; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>JavaScript Test Page</h1>
        <button onclick="testFunction()">Test JavaScript</button>
        <button onclick="testFetch()">Test Fetch API</button>
        <div id="result">Click a button to test JavaScript functionality</div>
        
        <script>
            function testFunction() {
                document.getElementById('result').innerHTML = 'JavaScript is working! ' + new Date().toLocaleTimeString();
                console.log('JavaScript executed successfully');
            }
            
            async function testFetch() {
                try {
                    const response = await fetch('/list-scripts');
                    const data = await response.json();
                    document.getElementById('result').innerHTML = 'Fetch API working! Found ' + data.scripts.length + ' scripts.';
                    console.log('Fetch API working:', data);
                } catch (error) {
                    document.getElementById('result').innerHTML = 'Fetch error: ' + error.message;
                    console.error('Fetch error:', error);
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/debug')
def debug():
    """Debug route to check file system"""
    try:
        carfac_dir = SCRIPT_BASE_PATH / 'carfac'
        all_carfac_files = list(carfac_dir.glob("*.py")) if carfac_dir.exists() else []
        mandarin_files = list(carfac_dir.glob("mandarin_*.py")) if carfac_dir.exists() else []
        
        target_file = carfac_dir / "mandarin_mu.py"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Debug Info</title></head>
        <body>
        <h2>Debug Information</h2>
        <p><strong>Base path (src):</strong> {SCRIPT_BASE_PATH}</p>
        <p><strong>Base path exists:</strong> {SCRIPT_BASE_PATH.exists()}</p>
        <p><strong>Carfac directory:</strong> {carfac_dir}</p>
        <p><strong>Carfac directory exists:</strong> {carfac_dir.exists()}</p>
        <p><strong>Target file:</strong> {target_file}</p>
        <p><strong>Target exists:</strong> {target_file.exists()}</p>
        <p><strong>Python executable:</strong> {PYTHON_EXECUTABLE}</p>
        <p><strong>Working command:</strong> {PYTHON_EXECUTABLE} carfac\\mandarin_mu.py</p>
        <p><strong>Working directory:</strong> {SCRIPT_BASE_PATH}</p>
        
        <h3>Files in carfac/ directory ({len(all_carfac_files)}):</h3>
        <ul>
        {''.join([f'<li>{f.name} (size: {f.stat().st_size} bytes)</li>' for f in all_carfac_files])}
        </ul>
        
        <h3>Mandarin files specifically ({len(mandarin_files)}):</h3>
        <ul>
        {''.join([f'<li>{f.name}</li>' for f in mandarin_files])}
        </ul>
        
        <h3>System Information:</h3>
        <ul>
        <li>OS: {os.name}</li>
        <li>Python version: {sys.version}</li>
        <li>Current working directory: {os.getcwd()}</li>
        </ul>
        
        <h3>Manual Command Test:</h3>
        <p>Your working manual command:</p>
        <code>cd {SCRIPT_BASE_PATH}</code><br>
        <code>python carfac\\mandarin_mu.py</code>
        
        <p><a href="/">Back to main page</a></p>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"Debug error: {e}"

# Create templates directory and HTML file
def create_template():
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pitchogram - Pronunciation Trainer</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      text-align: center;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      min-height: 100vh;
      margin: 0;
    }
    .container {
      display: flex;
      justify-content: space-around;
      margin-top: 30px;
      flex-wrap: wrap;
    }
    .language {
      width: 30%;
      min-width: 300px;
      margin: 10px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 15px;
      padding: 20px;
      backdrop-filter: blur(10px);
    }
    svg {
      width: 100%;
      height: 120px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.05);
    }
    h1 {
      font-size: 2.5em;
      margin-bottom: 10px;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2 {
      margin-bottom: 15px;
      font-size: 1.5em;
    }
    button {
      margin: 5px;
      padding: 10px 20px;
      font-size: 14px;
      cursor: pointer;
      background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
      color: white;
      border: none;
      border-radius: 25px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      min-width: 120px;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      background: linear-gradient(45deg, #FF5252, #26C6DA);
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    .status {
      margin-top: 20px;
      padding: 15px;
      border-radius: 8px;
      display: none;
      max-width: 500px;
      margin-left: auto;
      margin-right: auto;
    }
    .success {
      background: rgba(76, 175, 80, 0.8);
    }
    .error {
      background: rgba(244, 67, 54, 0.8);
    }
    .loading {
      background: rgba(255, 193, 7, 0.8);
    }
    .instructions {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 15px;
      margin: 20px auto;
      max-width: 600px;
      backdrop-filter: blur(10px);
    }
    .debug-links {
      margin-top: 20px;
    }
    .debug-links a {
      color: #FFD700;
      text-decoration: none;
      margin: 0 10px;
    }
    .debug-links a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <h1>Pitchogram - Pronunciation Trainer</h1>
  <div class="instructions">
    <strong>Click any button below to start the pronunciation trainer for that word or phrase.</strong><br>
    The training window will open automatically in a new console window.
  </div>
  
  <div class="container">
    <div class="language">
      <h2>Mandarin</h2>
      <svg viewBox="0 0 100 80">
        <defs>
          <linearGradient id="mandarin-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#ff6b6b;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#4ecdc4;stop-opacity:1" />
          </linearGradient>
        </defs>
        <path d="M10,70 Q30,20 50,40 Q70,10 90,60" stroke="url(#mandarin-grad)" fill="transparent" stroke-width="3"/>
        <circle cx="10" cy="70" r="2" fill="white"/>
        <circle cx="90" cy="60" r="2" fill="white"/>
      </svg>
      <button onclick="runScript('mandarin_mu.py', 'mǔ (3rd tone)')">mǔ</button>
      <button onclick="runScript('mandarin_mi.py', 'mí (2nd tone)')">mí</button>
      <button onclick="runScript('mandarin_me.py', 'mè (4th tone)')">mè</button>
      <button onclick="runScript('mandarin_ma.py', 'mā (1st tone)')">mā</button>
      <button onclick="runScript('mandarin_sentence.py', 'Mā zài jiā')">Mā zài jiā</button>
    </div>
    
    <div class="language">
      <h2>Vietnamese</h2>
      <svg viewBox="0 0 100 80">
        <defs>
          <linearGradient id="vietnamese-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#a8e6cf;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#ff8a80;stop-opacity:1" />
          </linearGradient>
        </defs>
        <path d="M10,40 Q25,15 40,40 Q55,65 70,40 Q85,15 90,40" stroke="url(#vietnamese-grad)" fill="transparent" stroke-width="3"/>
        <circle cx="10" cy="40" r="2" fill="white"/>
        <circle cx="90" cy="40" r="2" fill="white"/>
      </svg>
      <button onclick="runScript('vietnamese_bu.py', 'bủ')">bủ</button>
      <button onclick="runScript('vietnamese_bi.py', 'bí')">bí</button>
      <button onclick="runScript('vietnamese_bo.py', 'bọ')">bọ</button>
      <button onclick="runScript('vietnamese_be.py', 'bê')">bê</button>
      <button onclick="runScript('vietnamese_ba.py', 'ba')">ba</button>
      <button onclick="runScript('vietnamese_sentence.py', 'Ba đọc sách.')">Ba đọc sách.</button>
    </div>
    
    <div class="language">
      <h2>Thai</h2>
      <svg viewBox="0 0 100 80">
        <defs>
          <linearGradient id="thai-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#ffd93d;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#ff6b9d;stop-opacity:1" />
          </linearGradient>
        </defs>
        <path d="M10,55 Q25,25 40,55 Q55,25 70,55 Q85,25 90,55" stroke="url(#thai-grad)" fill="transparent" stroke-width="3"/>
        <circle cx="10" cy="55" r="2" fill="white"/>
        <circle cx="90" cy="55" r="2" fill="white"/>
      </svg>
      <button onclick="runScript('thai_mu.py', 'มู (mu)')">มู (mu)</button>
      <button onclick="runScript('thai_le.py', 'เล (le)')">เล (le)</button>
      <button onclick="runScript('thai_ko.py', 'โค (ko)')">โค (ko)</button>
      <button onclick="runScript('thai_ni.py', 'นิ (ni)')">นิ (ni)</button>
      <button onclick="runScript('thai_ta.py', 'ตา (ta)')">ตา (ta)</button>
      <button onclick="runScript('thai_sentence.py', 'ฉันมีตา. (I have eyes.)')">ฉันมีตา.</button>
    </div>
  </div>

  <div id="status" class="status">
    <p id="status-message"></p>
  </div>

  <div class="debug-links">
    <a href="/test-js">Test JavaScript</a> |
    <a href="/debug">Debug Info</a> |
    <a href="/list-scripts">List Scripts</a>
  </div>

  <script>
    function showStatus(message, type) {
      var statusDiv = document.getElementById('status');
      var statusMessage = document.getElementById('status-message');
      
      statusMessage.innerHTML = message;
      statusDiv.className = 'status ' + (type || 'success');
      statusDiv.style.display = 'block';
      
      if (type !== 'loading') {
        setTimeout(function() {
          statusDiv.style.display = 'none';
        }, 5000);
      }
    }

    function runScript(scriptName, displayText) {
      var button = event.target;
      var originalText = button.textContent;
      
      // Disable button and show loading
      button.disabled = true;
      button.textContent = 'Starting...';
      showStatus('Starting ' + displayText + ' pronunciation trainer...', 'loading');
      
      // Use XMLHttpRequest for better compatibility
      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/run-script', true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          // Re-enable button
          button.disabled = false;
          button.textContent = originalText;
          
          if (xhr.status === 200) {
            try {
              var result = JSON.parse(xhr.responseText);
              if (result.success) {
                showStatus('Successfully started ' + displayText + ' trainer! Check for a new window.', 'success');
              } else {
                showStatus('Error: ' + result.message, 'error');
              }
            } catch (e) {
              showStatus('Error parsing response: ' + e.message, 'error');
            }
          } else {
            showStatus('Network error: HTTP ' + xhr.status, 'error');
          }
        }
      };
      
      xhr.onerror = function() {
        button.disabled = false;
        button.textContent = originalText;
        showStatus('Network error: Request failed', 'error');
      };
      
      var data = JSON.stringify({ 
        script: scriptName,
        display_text: displayText
      });
      
      xhr.send(data);
    }

    // Check server connection on load
    window.addEventListener('load', function() {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', '/list-scripts', true);
      xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
          if (xhr.status === 200) {
            showStatus('Connected to server successfully!', 'success');
          } else {
            showStatus('Server connection failed. Status: ' + xhr.status, 'error');
          }
        }
      };
      xhr.send();
    });
  </script>
</body>
</html>'''
    
    with open(templates_dir / 'pitchogram.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    # Create the template file
    create_template()
    
    print("Flask Pitchogram Server")
    print("=" * 30)
    print(f"Script directory: {SCRIPT_BASE_PATH}")
    print(f"Directory exists: {SCRIPT_BASE_PATH.exists()}")
    print(f"Carfac directory: {SCRIPT_BASE_PATH / 'carfac'}")
    print(f"Carfac exists: {(SCRIPT_BASE_PATH / 'carfac').exists()}")
    print(f"Target script: {SCRIPT_BASE_PATH / 'carfac' / 'mandarin_mu.py'}")
    print(f"Target exists: {(SCRIPT_BASE_PATH / 'carfac' / 'mandarin_mu.py').exists()}")
    print(f"Server will start at: http://localhost:5000")
    print("\nDebug URLs:")
    print("  Main app: http://localhost:5000")
    print("  Test JS:  http://localhost:5000/test-js")
    print("  Debug:    http://localhost:5000/debug")
    print("\nThis Flask app will execute the equivalent of:")
    print(f"  cd {SCRIPT_BASE_PATH}")
    print("  python carfac\\mandarin_mu.py")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 30)
    
    # Run the Flask app
    app.run(debug=False, host='localhost', port=5000)
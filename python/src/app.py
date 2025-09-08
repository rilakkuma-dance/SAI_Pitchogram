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

# Configuration - Flexible path that works for everyone
# Get the directory where this app.py file is located
SCRIPT_BASE_PATH = Path(__file__).parent.absolute()
PYTHON_EXECUTABLE = "python"

# Alternative: You can also set it relative to current working directory
# SCRIPT_BASE_PATH = Path.cwd()

# Or use environment variable for even more flexibility
# SCRIPT_BASE_PATH = Path(os.environ.get('CARFAC_BASE_PATH', Path(__file__).parent.absolute()))

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
    return render_template('instruction.html')

@app.route('/trainer')
def trainer():
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

if __name__ == '__main__':
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
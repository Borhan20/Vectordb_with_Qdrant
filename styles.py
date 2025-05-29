# styles.py

from nicegui import ui

def apply_custom_styles():
    ui.add_head_html("""
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .upload-zone {
            border: 2px dashed #667eea;
            border-radius: 12px;
            transition: all 0.3s ease;
            background: linear-gradient(45deg, #f8f9ff, #e8efff);
        }
        .upload-zone:hover {
            border-color: #4f46e5;
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(103, 126, 234, 0.15);
        }
        .search-card {
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            border: none;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        .result-card {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border-left: 4px solid #667eea;
        }
        .result-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        }
        .stats-card {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .typing-animation {
            overflow: hidden;
            border-right: 2px solid #667eea;
            white-space: nowrap;
            animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
        }
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: #667eea; }
        }
        .upload-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }
        .upload-content {
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }
    </style>
    """)

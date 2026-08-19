import os
import uuid
import cv2
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from palm_analysis.preprocessing import validate_palm_image, preprocess_image
from palm_analysis.hand_detection import detect_hand
from palm_analysis.palm_detection import extract_palm_roi
from palm_analysis.line_detection import detect_palm_lines
from palm_analysis.feature_extraction import extract_features
from palm_analysis.interpretation import generate_interpretation
from database.db import (
    init_db, create_reading, get_reading, get_parsed_reading, update_reading_analysis,
    create_user, get_user_by_username_or_email, get_user_by_id, list_user_readings
)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Authentication required. Please sign in.'}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['DATABASE_PATH']), exist_ok=True)

    # Initialize SQLite database
    init_db()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if session.get('user_id'):
            return redirect(url_for('index'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')

            if not username or not email or not password:
                return render_template('register.html', error='All fields are required.')

            if len(password) < 6:
                return render_template('register.html', error='Password must be at least 6 characters.')

            password_hash = generate_password_hash(password)
            success, result = create_user(username, email, password_hash)

            if not success:
                return render_template('register.html', error=result)

            session['user_id'] = result
            session['username'] = username.lower()
            return redirect(url_for('upload'))

        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if session.get('user_id'):
            return redirect(url_for('index'))

        if request.method == 'POST':
            identifier = request.form.get('identifier', '').strip()
            password = request.form.get('password', '')

            if not identifier or not password:
                return render_template('login.html', error='Please enter your username/email and password.')

            user = get_user_by_username_or_email(identifier)
            if not user or not check_password_hash(user['password_hash'], password):
                return render_template('login.html', error='Invalid username/email or password.')

            session['user_id'] = user['id']
            session['username'] = user['username']

            next_url = request.args.get('next') or url_for('upload')
            return redirect(next_url)

        return render_template('login.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    @app.route('/history')
    @login_required
    def history():
        readings = list_user_readings(session['user_id'], limit=20)
        return render_template('history.html', readings=readings)

    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    def upload():
        if request.method == 'POST':
            if 'palm_image' not in request.files:
                return jsonify({'success': False, 'error': 'No file part in request.'}), 400

            file = request.files['palm_image']
            is_valid, msg, cv_img, width, height = validate_palm_image(file)

            if not is_valid:
                return jsonify({'success': False, 'error': msg}), 400

            # Generate unique reading ID
            reading_id = str(uuid.uuid4())
            original_filename = f"{reading_id}_original.jpg"
            processed_filename = f"{reading_id}_processed.jpg"

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

            # Preprocess image
            resized_bgr, enhanced_gray, enhanced_bgr = preprocess_image(cv_img)

            # Save images to disk
            cv2.imwrite(original_path, resized_bgr)
            cv2.imwrite(processed_path, enhanced_bgr)

            # Persist reading record in SQLite database
            create_reading(
                reading_id=reading_id,
                image_path=original_filename,
                processed_image_path=processed_filename,
                user_id=session.get('user_id')
            )

            return jsonify({
                'success': True,
                'message': 'Image uploaded and validated successfully.',
                'reading_id': reading_id,
                'redirect_url': url_for('analysis_page', reading_id=reading_id),
                'dimensions': {'width': width, 'height': height}
            })

        return render_template('upload.html')

    @app.route('/analysis/<reading_id>')
    @login_required
    def analysis_page(reading_id):
        reading = get_reading(reading_id)
        if not reading:
            return render_template('index.html', error='Reading not found.'), 404
        return render_template('analysis.html', reading_id=reading_id)

    @app.route('/api/analyze/<reading_id>', methods=['POST'])
    @login_required
    def run_analysis_api(reading_id):
        reading = get_reading(reading_id)
        if not reading:
            return jsonify({'success': False, 'error': 'Reading record not found.'}), 404

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], reading['image_path'])
        if not os.path.exists(original_path):
            return jsonify({'success': False, 'error': 'Original uploaded image file missing.'}), 404

        try:
            cv_img = cv2.imread(original_path)
            if cv_img is None:
                return jsonify({'success': False, 'error': 'Unable to read image for analysis.'}), 500

            # 1. MediaPipe / OpenCV Hand Detection
            hand_data = detect_hand(cv_img)

            # 2. Extract & Align Palm ROI
            palm_roi, roi_bbox = extract_palm_roi(cv_img, hand_data)

            # 3. Detect Palm Crease Lines
            line_data = detect_palm_lines(palm_roi, hand_type=hand_data.get('hand_type', 'Unknown'))

            # 4. Extract Structured Feature Vector
            features = extract_features(hand_data, line_data)

            # 5. Generate Traditional Palmistry Interpretation
            interpretation = generate_interpretation(features)

            # 6. Save Overlay Image
            overlay_filename = f"{reading_id}_overlay.jpg"
            overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
            cv2.imwrite(overlay_path, line_data['overlay_image'])

            # 7. Update DB Record
            update_reading_analysis(
                reading_id=reading_id,
                overlay_image_path=overlay_filename,
                hand_type=hand_data.get('hand_type', 'Unknown'),
                palm_features=features,
                line_features=line_data.get('lines', {}),
                interpretation=interpretation
            )

            return jsonify({
                'success': True,
                'reading_id': reading_id,
                'redirect_url': url_for('result_page', reading_id=reading_id)
            })

        except Exception as e:
            return jsonify({'success': False, 'error': f'Analysis pipeline failed: {str(e)}'}), 500

    @app.route('/result/<reading_id>')
    @login_required
    def result_page(reading_id):
        reading = get_parsed_reading(reading_id)
        if not reading:
            return render_template('index.html', error='Reading not found.'), 404

        # If analysis has not been executed yet, execute now
        if not reading.get('interpretation'):
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], reading['image_path'])
            if os.path.exists(original_path):
                cv_img = cv2.imread(original_path)
                hand_data = detect_hand(cv_img)
                palm_roi, _ = extract_palm_roi(cv_img, hand_data)
                line_data = detect_palm_lines(palm_roi, hand_type=hand_data.get('hand_type', 'Unknown'))
                features = extract_features(hand_data, line_data)
                interpretation = generate_interpretation(features)

                overlay_filename = f"{reading_id}_overlay.jpg"
                overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
                cv2.imwrite(overlay_path, line_data['overlay_image'])

                update_reading_analysis(
                    reading_id=reading_id,
                    overlay_image_path=overlay_filename,
                    hand_type=hand_data.get('hand_type', 'Unknown'),
                    palm_features=features,
                    line_features=line_data.get('lines', {}),
                    interpretation=interpretation
                )
                reading = get_parsed_reading(reading_id)

        return render_template('result.html', reading=reading)

    @app.route('/api/reading/<reading_id>')
    @login_required
    def get_reading_api(reading_id):
        reading = get_parsed_reading(reading_id)
        if not reading:
            return jsonify({'success': False, 'error': 'Reading not found'}), 404
        return jsonify({'success': True, 'reading': reading})

    @app.route('/api/chat', methods=['POST'])
    @login_required
    def chat_api():
        data = request.get_json() or {}
        reading_id = data.get('reading_id')
        question = data.get('question')

        if not reading_id or not question:
            return jsonify({'success': False, 'error': 'Missing reading_id or question.'}), 400

        reading = get_parsed_reading(reading_id)
        if not reading:
            return jsonify({'success': False, 'error': 'Reading not found.'}), 404

        from palm_analysis.chat import answer_user_question
        answer = answer_user_question(reading, question)

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/privacy')
    def privacy():
        return render_template('privacy.html')

    # Static file serving for uploads, processed, and results
    @app.route('/uploads/<path:filename>')
    def serve_upload(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/processed/<path:filename>')
    def serve_processed(filename):
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

    @app.route('/results/<path:filename>')
    def serve_results(filename):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)




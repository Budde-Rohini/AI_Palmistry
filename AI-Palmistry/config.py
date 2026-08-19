import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ai-palmistry-secret-key-2026')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'palmistry.db')
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

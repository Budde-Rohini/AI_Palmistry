import os
import sys
import uuid
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from database.db import init_db, get_user_by_username_or_email

class TestAuthentication(unittest.TestCase):

    def setUp(self):
        init_db()
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_registration_and_login_flow(self):
        unique_username = f"user_{uuid.uuid4().hex[:6]}"
        unique_email = f"{unique_username}@example.com"
        password = "securepassword123"

        # 1. Register new user
        reg_res = self.client.post('/register', data={
            'username': unique_username,
            'email': unique_email,
            'password': password
        }, follow_redirects=True)

        self.assertEqual(reg_res.status_code, 200)
        user = get_user_by_username_or_email(unique_username)
        self.assertIsNotNone(user)
        self.assertEqual(user['email'], unique_email)

        # 2. Logout
        self.client.get('/logout')

        # 3. Login with credentials
        login_res = self.client.post('/login', data={
            'identifier': unique_username,
            'password': password
        }, follow_redirects=True)

        self.assertEqual(login_res.status_code, 200)

    def test_protected_route_redirect(self):
        # Without logging in, /upload should redirect to /login
        res = self.client.get('/upload')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers['Location'])

if __name__ == '__main__':
    unittest.main()

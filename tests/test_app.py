import pytest
from app import app  # Assuming your Flask app is named "app" in app.py

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome" in response.data  # Change according to your actual homepage content

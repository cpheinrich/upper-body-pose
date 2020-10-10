from starlette.testclient import TestClient
from server import app
import os

client = TestClient(app)


def test_hello_world():
    endpoint = '/'
    r = client.get(endpoint)
    assert r.status_code == 200


def test_api():
    img_path = './serena.png'
    endpoint = '/predict'
    files = {'image': open(img_path, 'rb')}
    r = client.post(endpoint, files=files)
    assert r.status_code == 200
    print(r.json())
    assert "nose" in r.json().keys()
    """
    out_path = './serena_output.png'
    with open(out_path, 'wb') as f:
        f.write(r.content)
    assert os.path.isfile(out_path)
    """


def test_image_api():
    img_path = './serena.png'
    endpoint = '/predict_image'
    files = {'image': open(img_path, 'rb')}
    r = client.post(endpoint, files=files)
    assert r.status_code == 200
    out_path = './serena_output.png'
    with open(out_path, 'wb') as f:
        f.write(r.content)
    assert os.path.isfile(out_path)

from fastapi import FastAPI, UploadFile, File
from predictor import Predictor
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import io
import os


model = Predictor()

# Add prediction endpoints
app = FastAPI()


class APIOutput(BaseModel):
    image: bytes


@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    """ Predicts the predictions of 25 pose keypoints for the upper-body"""
    # Request -> PIL Image
    contents = await image.read()
    imgdata = io.BytesIO(contents)
    img = Image.open(imgdata)
    index = model.predictions % model.cache_size
    input_path = '/tmp/input_{}.png'.format(index)
    output_path = '/tmp/output_{}.png'.format(index)
    img.save(input_path)
    landmarks = model.predict(input_path=input_path, output_path=output_path)
    model.predictions += 1
    return landmarks


@app.post('/predict_image', response_model=APIOutput)
async def predict(image: UploadFile = File(...)):
    """ Returns the original image annotated by the pose keypoints"""
    # Request -> PIL Image
    contents = await image.read()
    imgdata = io.BytesIO(contents)
    img = Image.open(imgdata)
    index = model.predictions % model.cache_size
    input_path = '/tmp/input_{}.png'.format(index)
    output_path = '/tmp/output_{}.png'.format(index)
    img.save(input_path)
    landmarks = model.predict(input_path=input_path, output_path=output_path)
    model.predictions += 1
    return StreamingResponse(io.BytesIO(open(output_path, 'rb').read()), media_type='image/png')


@app.post('/predict_from_video')
async def predict(video: UploadFile = File(...), first_frame_index: int = 0):
    """ Predicts the predictions of 25 pose keypoints for the upper-body"""
    # Request -> PIL Image
    contents = await video.read()
    index = model.predictions % model.cache_size
    input_path = '/tmp/input_{}.avi'.format(index)

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(input_path):
        os.remove(input_path)

    out_file = open(input_path, "wb")  # open for [w]riting as [b]inary
    out_file.write(contents)
    out_file.close()

    landmarks = model.predict_from_video(
        input_path=input_path, first_frame_index=first_frame_index)
    model.predictions += 1
    return landmarks


@app.get('/')
def hello_world():
    return 'Hello world'

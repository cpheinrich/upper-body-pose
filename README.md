# Upper body pose tracking

## Getting started

Requires python 3.6+

Install dependencies

`pip install -r requirements.txt`

## Running inference

To run on a single video:

`python predict.py --from_video <video-path>`

To run on a directory of images

`python predict.py --from_dir <video-dir-path>`

After running inference, there should be two additional files for each input video. One file `*_landmarks.csv` contains the decoded pose landmarks stored in a csv file. The other file `*_network_output.mp4` contains the original video annotated with the landmark information.

NOTE: Inference should be fast and use very little memory, so you should easily be able to run this on your laptop

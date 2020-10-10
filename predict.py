from predictor import Predictor
import argparse
import os
import json
import pandas as pd

model = Predictor()

VIDEO_EXTENSIONS = ['.mov', '.mp4', '.wmv', '.flv',
                    '.avi', '.mpeg', '.mpg', '.webm', '.ogg']

VIDEO_SUFFIX = '_network_output.mp4'


def is_input_video(path):
    name, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in VIDEO_EXTENSIONS and VIDEO_SUFFIX not in name:
        return True
    else:
        return False


def run_inference(video_path):
    name, ext = os.path.splitext(video_path)
    output_path = video_path.replace(ext, '_landmarks.csv')
    output_video_path = video_path.replace(ext, VIDEO_SUFFIX)
    landmarks = model.predict_from_video(
        input_path=video_path, decoding='dataframe', output_video=output_video_path)

    landmarks.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Set global log level
    parser = argparse.ArgumentParser(
        description='Demonstration of landmarks localization.')
    parser.add_argument('--from_video', type=str,
                        help='Use this video path instead of webcam')
    parser.add_argument('--from_dir', type=str,
                        help='Path to a directory of videos')

    args = parser.parse_args()
    if args.from_video is not None:
        run_inference(args.from_video)
    elif args.from_dir is not None:
        videos = os.listdir(args.from_dir)
        for video in videos:
            if is_input_video(video):
                video_path = os.path.join(args.from_dir, video)
                run_inference(video_path)
            else:
                print("{} is not a valid video format. Skipping".format(video))
    else:
        raise NotImplementedError(
            """You must supply a video path to the --from_video argument or a path to a directory of videos with the  --from_dir argument to run inference!""")

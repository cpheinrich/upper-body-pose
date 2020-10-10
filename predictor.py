from paqr.predictor import PredictorInterface
import mediapipe as mp
import cv2
import time
import sys
import pandas as pd


class Predictor(PredictorInterface):
    def __init__(self):
        # Initialize your predictor
        self.pose_tracker = mp.examples.UpperBodyPoseTracker()
        self.predictions = 0
        self.cache_size = 25

    def decode_landmarks(self, landmarks: list, flatten: bool = True) -> dict:
        keys = ['nose', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_ear',
                'left_ear', 'mouth_right', 'mouth_left', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist',
                'right_pinky_1', 'left_pinky_1', 'right_index_1', 'left_index_1', 'right_thumb_2', 'left_thumb_2', 'right_hip', 'left_hip']
        output = {}
        for index, key in enumerate(keys):
            data_point = landmarks[index]
            if flatten:
                output['{}_x'.format(key)] = data_point.x
                output['{}_y'.format(key)] = data_point.y
                output['{}_z'.format(key)] = data_point.z
                output['{}_visibility'.format(key)] = data_point.visibility
            else:
                landmark = {}
                landmark['x'] = data_point.x
                landmark['y'] = data_point.y
                landmark['z'] = data_point.z
                landmark['visibility'] = data_point.visibility
                output[key] = landmark
        return output

    def predict(self, input_path: str, output_path: str):
        # Make a prediction
        pose_landmarks, _ = self.pose_tracker.run(
            input_file=input_path, output_file=output_path)
        return self.decode_landmarks(pose_landmarks.landmark)

    def predict_from_video(self, input_path: str, first_frame_index: int = 0, max_frame_count: int = None, decoding: str = 'json', output_video: str = ''):
        """ Runs inference on video, and decodes landmarks either as 'dataframe' or 'json' """
        # Make a prediction
        cap = cv2.VideoCapture(input_path)
        start_time = time.time()
        print(
            'Press Esc within the output image window to stop the run, or let it '
            'self terminate after 30 seconds.')
        if decoding == 'json':
            all_landmarks = {}
        elif decoding == 'dataframe':
            all_landmarks = pd.DataFrame()
        else:
            raise NotImplementedError

        frame_index = first_frame_index
        frame_count = 0
        if max_frame_count is None:
            max_frame_count = sys.maxsize

        video_out = None
        while cap.isOpened() and frame_count < max_frame_count:
            print("Frame count: {}, frame index: {}".format(
                frame_count, frame_index))
            success, input_frame = cap.read()
            if not success:
                break
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame.flags.writeable = False
            frame_landmarks, output_frame = self.pose_tracker._run_graph(
                input_frame)

            print("output frame shape", output_frame.shape)
            if output_video != '' and video_out is None:
                height, width, layers = output_frame.shape
                size = (width, height)
                video_out = cv2.VideoWriter(
                    output_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
            if video_out is not None:
                video_out.write(output_frame)

            if decoding == 'json':
                if frame_landmarks is not None:
                    all_landmarks[frame_index] = self.decode_landmarks(
                        frame_landmarks.landmark)
            else:
                if frame_landmarks is not None:
                    output = self.decode_landmarks(
                        frame_landmarks.landmark, flatten=True)
                    output['frame_index'] = int(frame_index)
                    all_landmarks = all_landmarks.append(
                        output, ignore_index=True)
            frame_index += 1
            frame_count += 1

        if video_out is not None:
            video_out.release()
        cap.release()
        cv2.destroyAllWindows()
        return all_landmarks

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--HEIGHT', type=int, default=288,
                    help='height of image input(default: 288)')
parser.add_argument('--WIDTH', type=int, default=512,
                    help='width of image input(default: 512)')
parser.add_argument('--load_weights', type=str, default="weights/TrackNet",
                    help='path to load pre-trained weights(default: weights/TrackNet)')
parser.add_argument('--frame_stack', type=int, default=3,
                    help='number of frames to be stacked(default: 3)')
parser.add_argument('--video_path', type=str, default='test/test.mp4',
                    help='video path to predict')

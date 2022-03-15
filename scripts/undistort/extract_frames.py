import argparse
from importmagic import import_from

with import_from('./'):
	from scripts.colmap2nerf import run_ffmpeg


def parse_args():
	parser = argparse.ArgumentParser(description="use ffmpeg to extract frames")
	parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--images", default="images", help="input path to the images")
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()
	run_ffmpeg(args)

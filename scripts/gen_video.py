import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from common import *
import pyngp as ngp # noqa


def render_frames(testbed, resolution, numframes, spp, fps, exposure=0, out_dir='./'):
	for i in tqdm(list(range(numframes)), unit="frames", desc="Rendering"):
		testbed.camera_smoothing = i > 0
		frame = testbed.render(resolution[0], resolution[1], spp, True, float(i)/numframes, float(i + 1)/numframes, fps, shutter_fraction=0.5)
		write_image(os.path.join(out_dir, "{}.jpg".format(i)), np.clip(frame * 2**exposure, 0.0, 1.0), quality=100)

def make_video(numframes, fps, resolution, out_dir='./'):
	writer = cv2.VideoWriter(os.path.join(out_dir, 'rendered.avi'),
                             cv2.VideoWriter_fourcc(*'XVID'),
                             fps,
                             resolution)

	for i in range(numframes):
		img = cv2.imread(os.path.join(out_dir, "{}.jpg".format(i)))
		writer.write(img)

	writer.release()


def parse_args():
	parser = argparse.ArgumentParser(description="Get frames for a video.")
	parser.add_argument("--load_snapshot", default="", help="Load this snapshot. recommended extension: .msgpack")
	parser.add_argument("--load_camera_path", default="", help="Load this camera path. recommended extension: .json")
	parser.add_argument("--out_dir", default="", help="Which directory to output frames/video to.")
	parser.add_argument("--spp", type=int, default=8, help="Number of samples per pixel in screenshots.")
	parser.add_argument("--width", type=int, default=1280, help="Resolution width.")
	parser.add_argument("--height", type=int, default=720, help="Resolution height.")
	parser.add_argument("--exposure", type=int, default=0, help="Exposure.")
	parser.add_argument("--seconds", type=int, default=5, help="Video length.")
	parser.add_argument("--fps", type=int, default=25, help="Frame rate.")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	mode = ngp.TestbedMode.Nerf
	testbed = ngp.Testbed(mode)
	testbed.load_snapshot(args.load_snapshot)
	testbed.background_color = [0.0, 0.0, 0.0, 1.0]
	testbed.shall_train = False
	testbed.load_camera_path(args.load_camera_path)
	numframes = int(args.seconds * args.fps)
	resolution = [args.width, args.height]
	os.makedirs(args.out_dir, exist_ok=True)

	render_frames(testbed, resolution, numframes,
				  spp=args.spp, fps=args.fps, exposure=args.exposure, out_dir=args.out_dir)
	
	make_video(numframes, args.fps, resolution, args.out_dir)

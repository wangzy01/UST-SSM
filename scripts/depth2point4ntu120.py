import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
import concurrent.futures

parser = argparse.ArgumentParser(description='Depth to Point Cloud')
parser.add_argument('--input', default='', type=str)
parser.add_argument('--output', default='', type=str)

args = parser.parse_args()

W = 512
H = 424
focal = 280

xx, yy = np.meshgrid(np.arange(W), np.arange(H))

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_video(video_path, output_dir):
    """ Process a single video directory and save the point clouds."""
    video_name = video_path.split('/')[-1]
    print(f"Processing: {video_name}")
    point_clouds = []

    for img_name in sorted(os.listdir(video_path)):
        img_path = os.path.join(video_path, img_name)
        img = imread(img_path)  # (H, W)

        depth_min = img[img > 0].min()
        depth_map = img

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]
        x = (x - W / 2) / focal * z
        y = (y - H / 2) / focal * z

        points = np.stack([x, y, z], axis=-1)
        point_clouds.append(points)

    output_file = os.path.join(output_dir, video_name + '.npz')
    np.savez_compressed(output_file, data=np.array(point_clouds, dtype=object))
    print(f"Finished: {video_name}")

def process_action(action):
    """Process all videos for a specific action."""
    for video_path in sorted(glob(f'{args.input}/*A0{action:02d}')):
        process_video(video_path, args.output)
    print(f'Action {action:02d} finished!')

def main():
    mkdir(args.output)
    actions = range(1, 121)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_action, actions)

if __name__ == "__main__":
    main()

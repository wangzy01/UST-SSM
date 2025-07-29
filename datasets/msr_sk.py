import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ipdb import set_trace as st


class MSRAction3D_SK(Dataset):
    def __init__(self, root, skeleton_root, frames_per_clip=54, step_between_clips=1, num_points=2048, train=True):
        super(MSRAction3D_SK, self).__init__()

        self.videos = []
        self.skeletons = []  
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                skeleton_file = video_name.replace('_sdepth.npz', '_skeleton3D.txt')
                skeleton_path = os.path.join(skeleton_root, skeleton_file)
                skeleton_data = self.load_skeleton(skeleton_path)
                self.skeletons.append(skeleton_data)
                
                label = int(video_name.split('_')[0][1:]) - 1
                self.labels.append(label)

                nframes = video.shape[0]

                for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                
                skeleton_file = video_name.replace('_sdepth.npz', '_skeleton3D.txt')
                skeleton_path = os.path.join(skeleton_root, skeleton_file)
                skeleton_data = self.load_skeleton(skeleton_path)
                self.skeletons.append(skeleton_data)
                
                label = int(video_name.split('_')[0][1:]) - 1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                    self.index_map.append((index, t))  # 0,1,2...30  #4478
                index += 1
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = 20

    def load_skeleton(self, file_path):
        skeleton_data = np.loadtxt(file_path).reshape(-1, 20, 4)  # (num_frames, 20, 4)
        return skeleton_data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        skeleton = self.skeletons[index]

        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        skeleton_clip = [skeleton[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]

        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300   
        skeleton_clip = np.array(skeleton_clip).astype(np.float32)
        skeleton_clip = skeleton_clip[:,:,:3]

        return (clip.astype(np.float32), np.array(skeleton_clip).astype(np.float32)), label, index



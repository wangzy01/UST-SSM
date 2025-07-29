import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ipdb import set_trace as st


Cross_Subject = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]


def clip_normalize(clip, epsilon=1e-6):
    pc = clip.view(-1, 3) 
    centroid = pc.mean(dim=0)  
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    
    if m < epsilon:
        m = epsilon
    
    clip = (clip - centroid) / m
    return clip

class NTU60Subject_SK(Dataset):
    def __init__(self, root, skeleton_root, meta, frames_per_clip=24, step_between_clips=2, num_points=2048, train=True):
        super(NTU60Subject_SK, self).__init__()

        self.videos = []
        self.labels = []
        self.skeleton_files = []
        self.index_map = []
        index = 0
        with open(meta, 'r') as f:
            for line in f:
                name, nframes_meta = line.strip().split()
                subject = int(name[9:12])
                nframes_meta = int(nframes_meta)

                if (train and subject in Cross_Subject) or (not train and subject not in Cross_Subject):
                    label = int(name[-3:]) - 1

                    skeleton_file = os.path.join(skeleton_root, name + '.npz')
                    if not os.path.exists(skeleton_file):
                        continue

                    video_file = os.path.join(root, name + '.npz')
                    if not os.path.exists(video_file):
                        continue


                    min_frames = nframes_meta
                    if min_frames < frames_per_clip * step_between_clips:
                        continue

                    nframes = min_frames

                    for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                        self.index_map.append((index, t))

                    self.labels.append(label)
                    self.videos.append(video_file)
                    self.skeleton_files.append(skeleton_file)
                    index += 1

        if len(self.labels) == 0:
            raise ValueError("No data found. Please check your dataset paths and meta file.")

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)


    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video_path = self.videos[index]
        skeleton_file = self.skeleton_files[index]
        video = np.load(video_path, allow_pickle=True)['data'] * 100
        label = self.labels[index]

        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        clip = [torch.tensor(p).float() for p in clip]  
        skeleton_clip = []

        try:
            with np.load(skeleton_file, allow_pickle=True) as data:
                skeleton_data = data['data']
        except Exception as e:
            print(f"Error loading skeleton file {skeleton_file}: {e}")
            raise

        for i in range(len(clip)):
            frame_idx = t + i * self.step_between_clips
            p = clip[i]
            s = torch.tensor(skeleton_data[frame_idx]).float()

            if p.shape[0] > self.num_points:
                r = torch.randperm(p.shape[0])[:self.num_points]
            else:
                repeat, residue = divmod(self.num_points, p.shape[0])
                r = torch.cat([torch.arange(p.shape[0]).repeat(repeat), 
                                torch.randperm(p.shape[0])[:residue]])
            
            clip[i] = p[r, :]
            skeleton_clip.append(s)
            
        clip = clip_normalize(torch.stack(clip))
        skeleton_clip = clip_normalize(torch.stack(skeleton_clip))

        if self.train:
            scales = torch.FloatTensor(3).uniform_(0.9, 1.1)
            clip = clip * scales

        return (clip.float(), skeleton_clip.float()), label, index


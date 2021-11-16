import numpy as np
import cv2
import os
import argparse
from networks import *
from skyboxengine import *
import torch


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SKYAR')
parser.add_argument('--net_G', type=str, default='coord_resnet50', help='net_G')
parser.add_argument('--ckptdir', type=str, default='./checkpoints_G_coord_resnet50', help='ckptdir')
parser.add_argument('--input_mode', type=str, default='video', help='input mode')
parser.add_argument('--out_size_w', type=int, default=384, help='output size width')
parser.add_argument('--out_size_h', type=int, default=192, help='output size height')
parser.add_argument('--input_folder', type=str, default='', help='input data folder')
parser.add_argument('--output_folder', type=str, default='', help='output data folder')


def GetFileNamesRecursive(rootPath, file_types = ['.jpg', '.png']) :
    if len(file_types) > 0:
        for file_type in file_types:
            file_type = file_type.lower()

    abs_file_paths = []
    for root, dirnames, filenames in os.walk(rootPath):
        for filename in filenames:
            if len(file_types) > 0:
                suffix = os.path.splitext(filename)[-1]
                suffix = suffix.lower()
                if suffix not in file_types:
                    continue

            abs_file_paths.append(os.path.join(root, filename))

    abs_file_paths = sorted(abs_file_paths)

    file_paths = []
    for file_path in abs_file_paths :
        file_path = file_path[len(rootPath) + 1:]
        file_paths.append(file_path)

    return file_paths, abs_file_paths


class SkyFilter():

    def __init__(self, args):

        self.ckptdir = args.ckptdir
        self.input_mode = args.input_mode
        self.input_folder = args.input_folder
        self.output_folder = args.output_folder

        self.out_size_w, self.out_size_h = args.out_size_w, args.out_size_h

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=args.net_G).to(device)
        self.load_model()

        if os.path.exists(self.output_folder) is False:
            os.makedirs(self.output_folder)


    def load_model(self):

        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'))
        # checkpoint = torch.load(os.path.join(self.ckptdir, 'last_ckpt.pt'))
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()


    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD


    def run_imgseq(self):

        print('running evaluation...')
        img_names, img_paths_full = GetFileNamesRecursive(self.input_folder)

        for idx in range(len(img_names)):
            img_HD = cv2.imread(img_paths_full[idx], cv2.IMREAD_COLOR)
            img_HD = self.cvtcolor_and_resize(img_HD)

            img = np.array(img_HD, dtype=np.float32)
            img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

            with torch.no_grad():
                G_pred = self.net_G(img.to(device))
                G_pred = G_pred[0, :].permute([1, 2, 0])
                G_pred = np.array(G_pred.detach().cpu())
                G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

                fpath = os.path.join(self.output_folder, img_names[idx])
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                cv2.imwrite(fpath, np.array(255.0 * G_pred, dtype=np.uint8))

            print('processing: %d / %d ...' % (idx, len(img_names)))


    def run_video(self):

        print('running evaluation...')
        video_names, video_paths_full = GetFileNamesRecursive(self.input_folder, ['.mp4'])

        for idx_video in range(len(video_names)):
            cap = cv2.VideoCapture(video_paths_full[idx_video])

            fpath = os.path.join(self.output_folder, video_names[idx_video])
            video_writer = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (self.out_size_w, self.out_size_h), False)
            m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            idx = 0
            while (1):
                ret, frame = cap.read()
                if ret:
                    img_HD = self.cvtcolor_and_resize(frame)

                    img = np.array(img_HD, dtype=np.float32)
                    img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

                    with torch.no_grad():
                        G_pred = self.net_G(img.to(device))
                        G_pred = G_pred[0, :].permute([1, 2, 0])
                        G_pred = np.array(G_pred.detach().cpu())
                        G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

                    video_writer.write(np.array(255.0 * G_pred, dtype=np.uint8))
                    print('processing: %d / %d (%d/%d)...' % (idx, m_frames, idx_video + 1, len(video_names)))

                    idx += 1
                else:  # if reach the last frame
                    break

            video_writer.release()


    def run(self):
        if self.input_mode == 'seq':
            self.run_imgseq()
        elif self.input_mode == 'video':
            self.run_video()
        else:
            print('wrong input_mode, select one in [seq, video')
            exit()


if __name__ == '__main__':

    parser = parser.parse_args()

    sf = SkyFilter(parser)
    sf.run()



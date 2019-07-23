from __future__ import unicode_literals
from configuration import cfg
import youtube_dl
import os
import glob
import multiprocessing as mp


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


SPORTS_FILE = cfg['SPORTS_1M_LIST']
SPORTS_DATASET_DIR = cfg['SPORTS_DATASET_DIR']
SPORTS_DATASET_CLASS_LABELS = cfg['SPORTS_DATASET_LABELS']
flabels = open(SPORTS_DATASET_CLASS_LABELS)
lines = flabels.readlines()
classes = {}
class_index = 0
for c in lines:
    classes[class_index] = c.split('\n')[0]
    class_index = class_index + 1
flabels.close()

mutex = mp.Lock()


def execution(l):
    ydl_opts = {
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
    }
    print(l)
    splits = l.split()
    video_url = splits[0]
    ids = splits[1].split(',')
    for i in ids:
        inti = int(i)
        video_id = video_url.split("=")[1]
        mutex.acquire()
        glob_res = glob.glob(SPORTS_DATASET_DIR + '/' + classes[inti] + '/' +
                             video_id)
        if len(glob_res) > 0:
            continue
        if not os.path.exists(SPORTS_DATASET_DIR + '/' + classes[inti]):
            os.mkdir(SPORTS_DATASET_DIR + '/' + classes[inti])
        mutex.release()
    ydl_opts['outtmpl'] = SPORTS_DATASET_DIR + '/' + classes[
        inti] + '/' + video_id + '.mp4'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            print('downloaded to' + ydl_opts['outtmpl'])
        except Exception as e:
            print('downloading ' + video_id + 'failed')


def main():
    pool = mp.Pool(mp.cpu_count())
    f = open(SPORTS_FILE, 'r')
    lines = f.readlines()
    pool.map(execution, [l for l in lines])
    pool.close()
    f.close()


if __name__ == "__main__":
    main()

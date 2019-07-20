from __future__ import unicode_literals
from configuration import cfg
import youtube_dl
import os


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


def main():

    ydl_opts = {
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
    }
    SPORTS_FILE = cfg['SPORTS_1M_LIST']
    SPORTS_DATASET_DIR = cfg['SPORTS_DATASET_DIR']
    SPORTS_DATASET_CLASS_LABELS = cfg['SPORTS_DATASET_LABELS']
    flabels = open(SPORTS_DATASET_CLASS_LABELS)
    lines = flabels.readlines()
    classes = {}
    class_index = 0
    for l in lines:
        classes[class_index] = l
        class_index = class_index + 1
    flabels.close()
    f = open(SPORTS_FILE, 'r')
    lines = f.readlines()
    for l in lines:
        print(l)
        splits = l.split()
        video_url = splits[0]
        ids = splits[1].split(',')
        for i in ids:
            if os.path.exists(SPORTS_DATASET_DIR + '/' + classes[i] + '/' +
                              video_id + '.mkv') or os.path.exists(
                                  SPORTS_DATASET_DIR + '/' + classes[i] + '/' +
                                  video_id + '.mp4') or os.path.exists(
                                      SPORTS_DATASET_DIR + '/' + classes[i] +
                                      '/' + video_id + '.webm'):
                continue
            if not os.path.exists(SPORTS_DATASET_DIR + '/' + classes[i]):
                os.mkdir(SPORTS_DATASET_DIR + '/' + classes[i])
        video_id = video_url.split("=")[1]
        ydl_opts['outtmpl'] = SPORTS_DATASET_DIR + '/' + classes[
            i] + '/' + video_id + '.mp4'
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                print('downloading ' + video_id + 'of class ' + classes[i] +
                      'failed')

    f.close()


if __name__ == "__main__":
    main()

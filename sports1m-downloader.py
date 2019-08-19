from __future__ import unicode_literals
from configuration import cfg
import youtube_dl
import os
import glob
import multiprocessing as mp
from shutil import copy
import datetime


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print_with_date(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print_with_date('Done downloading, now converting ...')


def print_with_date(s):
    print(str(datetime.datetime.now()) + ': ' + s + '--' +
          str(mp.current_process()))


#SPORTS_FILE = cfg['SPORTS_1M_LIST_SUBSET_SORTED']
SPORTS_FILE = cfg['SPORTS_1M_TEST_LIST']
#SPORTS_FILE_SUBSET = cfg['SPORTS_1M_LIST_SUBSET']
SPORTS_FILE_SUBSET = cfg['SPORTS_1M_TEST_LIST_SUBSET']
#SPORTS_FILE_SUBSET_SORTED = cfg['SPORTS_1M_LIST_SUBSET_SORTED']
SPORTS_FILE_SUBSET_SORTED = cfg['SPORTS_1M_TEST_LIST_SUBSET_SORTED']
SPORTS_DATASET_DIR = cfg['SPORTS_DATASET_DIR']
SPORTS_DATASET_CLASS_LABELS = cfg['SPORTS_DATASET_LABELS']
flabels = open(SPORTS_DATASET_CLASS_LABELS)
lines = flabels.readlines()
classes = {}
class_index = 0
manager = mp.Manager()
counters = manager.dict()
for c in lines:
    classes[class_index] = c.split('\n')[0]
    counters[class_index] = []
    class_index = class_index + 1
flabels.close()

mutex = mp.Lock()


def extract_info(l):
    ydl_opts = {'quiet': True, 'skip_download': True, 'logger': MyLogger()}
    print_with_date(l)
    splits = l.split()
    video_url = splits[0]
    ids = splits[1].split(',')
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url)
        except Exception as e:
            print_with_date('downloading ' + video_url + 'failed')
            return
    duration = int(info.get('duration') or 0)

    if duration > 0 and duration <= 90:
        mutex.acquire()
        print_with_date("------ACQUIRED LOCK-----")
        write = False
        for i in ids:
            inti = int(i)
            if inti in counters and len(counters[inti]) < 150:
                write = True
            elif not inti in counters:
                counters[inti] = []
                write = True
        if write:
            for i in ids:
                inti = int(i)
                if not inti in counters:
                    counters[inti] = []
                counters[inti].append(l)
            print_with_date('PUSHED ' + video_url + " TO DATA OBJ")
        mutex.release()
        print_with_date("------RELEASED LOCK-----")
    print_with_date('Video ' + video_url + ' is ' + str(duration) + ' long')
    print_with_date(str(counters))


def execution(l):
    ydl_opts = {
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
    }
    print_with_date(l)
    splits = l.split()
    video_url = splits[0]
    ids = splits[1].split(',')
    src = ''
    dsts = []
    copy = False
    for i in ids:
        inti = int(i)
        video_id = video_url.split("=")[1]
        glob_res = glob.glob(SPORTS_DATASET_DIR + '/' + classes[inti] + '/' +
                             video_id + '*')
        if len(glob_res) > 0:
            print_with_date(SPORTS_DATASET_DIR + '/' + classes[inti] + '/' +
                            video_id + ' already exists as ' + glob_res[0])
            src = glob_res[0]
            copy = True
            continue
        if not os.path.exists(SPORTS_DATASET_DIR + '/' + classes[inti]):
            os.mkdir(SPORTS_DATASET_DIR + '/' + classes[inti])

        dsts.append(SPORTS_DATASET_DIR + '/' + classes[inti] + '/')
    if copy:
        for dst in dsts:
            if dst == src:
                continue
            copy(src, dst)
            print_with_date('copied from ' + src + ' to ' + dst)
        return src
    ydl_opts['outtmpl'] = SPORTS_DATASET_DIR + '/' + classes[
        inti] + '/' + video_id + '.mp4'
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            print_with_date('downloaded to' + ydl_opts['outtmpl'])
        except Exception as e:
            print_with_date('downloading ' + video_id + 'failed')
    return ydl_opts['outtmpl']


def sort_file(lines):
    data = {}
    for l in lines:
        splits = l.split()
        ids = splits[1].split(',')
        for i in ids:
            inti = int(i)
            if not inti in data:
                data[inti] = []
            data[inti].append(l)
    return data


def write_data_obj(data, output):
    for k in data:
        for i in data[k]:
            output.write(i)


def main():
    print_with_date('available cpus ' + str(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    print_with_date('initialised pool')
    f = open(SPORTS_FILE, 'r')
    print_with_date('readlines....')
    lines = f.readlines()
    print_with_date('staring forks.........')
    pool.map(extract_info, [l for l in lines])
    pool.close()
    f.close()
    outfile = open(SPORTS_FILE_SUBSET, 'a+')
    write_data_obj(counters, outfile)
    # f = open(SPORTS_FILE_SUBSET, 'r')
    # lines = f.readlines()
    # print_with_date('SORTING FILE')
    # counters = sort_file(lines)
    # print_with_date('DONE SORTING')
    # print_with_date('WRITING FILE')
    # outfile = open(SPORTS_FILE_SUBSET_SORTED, 'a+')
    # write_data_obj(counters, outfile)
    print_with_date('DONE WRITING FILE')
    outfile.close()
    print(counters)


if __name__ == "__main__":
    main()

from PIL import Image
import os
import json
import logging
import shutil
import csv

# 跟踪训练进度的时钟类
class TrainClock(object):
    # 初始化时钟:轮次 批次 步数
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    # 增加批次和步数
    def tick(self):
        self.minibatch += 1
        self.step += 1

    # 增加 epoch，并将 minibatch 重置为零
    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    # 创建一个保存当前时钟状态的字典
    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    # 根据提供的字典恢复时钟状态
    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


# 用于记录实验结果的表格类
class Table(object):
    # 初始化表格对象
    def __init__(self, filename):
        '''
        create a table to record experiment results that can be opened by excel
        :param filename: using '.csv' as postfix
        '''
        assert '.csv' in filename
        self.filename = filename

    # 合并两个表格的表头
    @staticmethod
    def merge_headers(header1, header2):
        #return list(set(header1 + header2))
        if len(header1) > len(header2):
            return header1
        else:
            return header2

    # 写入一个条目到表格中，参数是一个类似于字典的有序对象，包含实验结果信息
    def write(self, ordered_dict):
        '''
        write an entry
        :param ordered_dict: something like {'name':'exp1', 'acc':90.5, 'epoch':50}
        :return:
        '''
        if os.path.exists(self.filename) == False:
            headers = list(ordered_dict.keys())
            prev_rec = None
        else:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                prev_rec = [row for row in reader]
            headers = self.merge_headers(headers, list(ordered_dict.keys()))

        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, headers)
            writer.writeheader()
            if not prev_rec == None:
                writer.writerows(prev_rec)
            writer.writerow(ordered_dict)


# 记录工作日志的日志记录器类
class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)


# 用于计算和存储平均值和当前值的类
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 将命令行参数保存到 JSON 文件中，以便记录实验配置
def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'params.json')

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def pad_to_16x(x):
    if x % 16 > 0:
        return x - x % 16 + 16
    return x


def pad_to_height(tar_height, img_height, img_width):
    scale = tar_height / img_height
    h = pad_to_16x(tar_height)
    w = pad_to_16x(int(img_width * scale))
    return h, w, scale


def test():
    pass


if __name__ == '__main__':
    test()

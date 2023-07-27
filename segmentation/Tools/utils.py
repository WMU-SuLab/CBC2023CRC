import time, os
from tensorboardX import SummaryWriter
import shutil
import torch
class LogMaker():
    def __init__(self, opt, config_file):

        self.use_centerness = opt.use_centerness
        # summary
        now = int(time.time())
        timeArray = time.localtime(now)
        log_folder = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
        self.log_folder = os.path.join('log', log_folder)
        self.result_folder = os.path.join(self.log_folder, 'running_result')
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.writer = SummaryWriter(self.log_folder)
        # record the hyperparameters
        shutil.copy(config_file, os.path.join(self.log_folder, 'config.py'))
        shutil.copy('./models/sw_gan.py', os.path.join(self.log_folder,'sw_gan.py'))
        shutil.copy('./models/network.py', os.path.join(self.log_folder, 'network.py'))
        shutil.copy('./models/layers.py', os.path.join(self.log_folder, 'layers.py'))
        shutil.copy('./loss.py', os.path.join(self.log_folder,'loss.py'))
    def write(self, step, losses):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, step)


    def draw_prediction(self, pred, targs, step):
        target_artery = targs[0:1, 0, :, :]

        pred_sigmoid = pred  # nn.Sigmoid()(pred)

        self.writer.add_image('bf', torch.cat([pred_sigmoid[0:1, 0, :, :], target_artery], dim=1), global_step=step)



    def print(self, losses, step):
        print_str = '{} step -'.format(step)
        for k, v in losses.items():
            print_str += ' {}:{:.4}'.format(k,v)
        print(print_str)

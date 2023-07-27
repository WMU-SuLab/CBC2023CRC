import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from models.network import GNet, set_requires_grad, VGGNet

from models import networks_gan
from loss import multiclassLoss,multidiceLoss,multiclassLossAV, CrossEntropyLossWithSmooth, L1LossWithLogits, vggloss, gradient_penalty, tripletMarginLoss_vggfea, centernessLoss, SmoothL1_weighted
import os, copy
from opt import get_patch_trad_5, modelEvalution
from collections import OrderedDict

class SW_GAN():

    def __init__(self, opt, isTrain=True):
        self.cfg = opt
        self.use_GAN = opt.use_GAN
        self.isTrain = isTrain
        self.use_cuda = opt.use_cuda
        # initilize all the loss names for print in each iteration
        self.get_loss_names(opt)

        # define networks (both generator and discriminator)
        self.netG = GNet(input_ch=opt.input_nc,
                        resnet = opt.use_network,
                        pretrained=True,
                        use_cuda=opt.use_cuda,
                        num_classes = opt.n_classes,
                        )
        if self.use_cuda:
            self.netG = self.netG.cuda()
        print(self.netG)

        if self.isTrain and opt.use_GAN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            tmp_num_classes_D = opt.num_classes_D

            self.netD = networks_gan.define_D(input_nc=opt.input_nc_D, ndf=opt.ndf,
                                              netD=opt.netD_type, n_layers_D=opt.n_layers_D,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid,
                                              init_type=opt.init_type, init_gain=opt.init_gain,
                                              gpu_ids=opt.gpu_ids,num_classes_D=tmp_num_classes_D,
                                              use_noise=opt.use_noise_input_D, use_dropout=opt.use_dropout_D)
            print(self.netD)
            self.netD.train()
            self.netG.train()



        if self.isTrain:

            # define loss functions

            self.criterionCE = nn.BCEWithLogitsLoss()
            self.criterionDICE = multidiceLoss()

            self.criterion = multiclassLoss()

            # initialize optimizers and scheduler.
            if opt.use_SGD:
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr,momentum = 0.9,weight_decay=5e-4)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G,step_size =opt.step_size, gamma=0.5)
            # self.scheduler_G = GradualWarmupScheduler(self.optimizer_G, multiplier=1, total_epoch=opt.step_size//20,
            #                                           after_scheduler=self.scheduler_G)
            if opt.use_GAN:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opt.step_size, gamma=opt.lr_decay_gamma)
                #self.scheduler_D = GradualWarmupScheduler(self.optimizer_D, multiplier=1, total_epoch=opt.step_size//20,after_scheduler=self.scheduler_D)


    def setup(self, opt, log_folder):
        # define the directory for logger
        self.log_folder = log_folder
        # mkdir for training result
        self.train_result_folder = os.path.join(self.log_folder, 'training_result')
        if not os.path.exists(self.train_result_folder):
            os.mkdir(self.train_result_folder)
        # load network
        if not self.isTrain or opt.use_pretrained_G:
            model_path = os.path.join(opt.model_path_pretrained_G, 'G_' + str(opt.model_step_pretrained_G) + '.pkl')
            pt = torch.load(model_path)
            model_static = self.netG.state_dict()
            pt_ = {k: v for k, v in pt.items() if k in model_static}

            model_static.update(pt_)

            self.netG.load_state_dict(model_static)

            print("Loading pretrained model for Generator from " + model_path)
            if opt.use_GAN:
                model_path_D = os.path.join(opt.model_path_pretrained_G, 'D_' + str(opt.model_step_pretrained_G) + '.pkl')
                self.netD.load_state_dict(torch.load(model_path_D))
                print("Loading pretrained model for Discriminator from " + model_path_D)

    def set_input(self, step, train_data=None, label_data=None, label_data_centerness=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """

        opt = self.cfg

        self.step = step

        data, label = get_patch_trad_5(opt.batch_size, opt.patch_size, train_data, label_data,

                                                    )


        self.data_input = torch.FloatTensor(data)
        self.label_input = torch.FloatTensor(label)


        if opt.use_cuda:
            self.data_input = self.data_input.cuda()
            self.label_input = self.label_input.cuda()

        # downsample the centerness scores maps and dilated images


        self.data_input = autograd.Variable(self.data_input)
        self.label_input = autograd.Variable(self.label_input)



        self.input = self.data_input


    def forward(self):

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.pre_target= self.netG(self.input)
        # sigmoid
        self.pre_target = torch.sigmoid(self.pre_target)

    def save_model(self):
        # save generator
        torch.save(self.netG.state_dict(), os.path.join(self.log_folder, 'G_' + str(self.step)+'.pkl'))
        torch.save(self.netG, os.path.join(self.log_folder, 'G_' + str(self.step)+'.pth'))
        # save discriminator
        if self.cfg.use_GAN:
            torch.save(self.netD.state_dict(), os.path.join(self.log_folder, 'D_' + str(self.step)+'.pkl'))
            torch.save(self.netD, os.path.join(self.log_folder, 'D_' + str(self.step)+'.pth'))
        print("save model to {}".format(self.log_folder))

    def log(self, logger):
        logger.draw_prediction(self.pre_target, self.label_input, self.step)

    def get_loss_names(self, opt):
        self.loss_names = []
        if opt.use_GAN:
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')


            self.loss_names.append('D')
            self.loss_names.append('G_GAN')

        self.loss_names.append('G_BCE')

        self.loss_names.append('G_DICE')

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_ret

    def test(self, result_folder):
        print("-----------start to test-----------")
        modelEvalution(self.step, self.netG.state_dict(),
                       result_folder,
                       use_cuda=self.cfg.use_cuda,
                       dataset=self.cfg.dataset,
                       input_ch=self.cfg.input_nc,
                       config=self.cfg,
                       strict_mode=True)
        print("---------------end-----------------")

    def backward_D(self, isBackward=True):

        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        opt = self.cfg

        # define the input of D
        real_input = torch.cat([self.data_input, self.label_input], dim=1)
        fake_input = torch.cat([self.data_input, self.pre_target], dim=1)

        pred_real = self.netD(real_input)
        pred_fake = self.netD(fake_input.detach())  # bs x ch x (HxW)  b,1,(h*w)

        # Compute loss
        self.loss_D = 0

        # for GT
        label_shape = [opt.batch_size, 1,pred_real.shape[2]]
        # 0, 1
        label_fake = torch.zeros(label_shape)
        label_real = torch.ones(label_shape)

        if opt.use_cuda:
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
        if opt.GAN_type == 'vanilla':
            self.loss_D_real = self.criterionCE(pred_real, label_real)
            self.loss_D_fake = self.criterionCE(pred_fake, label_fake)

            self.loss_D = (self.loss_D_real + self.loss_D_fake)
            self.loss_D = self.loss_D * opt.lambda_GAN_D  # loss_D_fake_shuffle

        # backward
        if isBackward:
            self.loss_D.backward()
    def backward_G(self, isBackward=True):

        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        opt = self.cfg

        # define input
        fake_input_cpy = torch.cat([self.data_input, self.pre_target], dim=1)
        self.loss_G = 0

        # GAN
        if opt.use_GAN:
            pred_fake = self.netD(fake_input_cpy)
            if opt.GAN_type == 'vanilla':
                ones_tensor = torch.ones([opt.batch_size,1, pred_fake.shape[2]])
                if opt.use_cuda:
                    ones_tensor = ones_tensor.cuda()
                self.loss_G_GAN = opt.lambda_GAN_G * self.criterionCE(pred_fake, ones_tensor)
            self.loss_G += self.loss_G_GAN

        # BCE
        self.loss_G_BCE = opt.lambda_BCE * self.criterion(self.pre_target, self.label_input)
        self.loss_G += self.loss_G_BCE
        self.loss_G_DICE = opt.lambda_DICE * self.criterionDICE(self.pre_target, self.label_input)
        self.loss_G += self.loss_G_DICE


        # centerness scores maps prediction

        if isBackward:
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        if self.use_GAN:
            # update D
            set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            self.scheduler_D.step(self.step)
        # update G
        if self.use_GAN:
            set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.scheduler_G.step(self.step)
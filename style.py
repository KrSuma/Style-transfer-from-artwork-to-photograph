# SOURCE: https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb

import time
import os
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
from tkinter import messagebox

import networks
import vgg19

import threading

class Stylizer():

    MODEL_PATH = 'models/vgg19/vgg_conv.pth'

    def __init__(self, style_weight=1000.0, content_weight=1.0):
        self.optimizer = optim.LBFGS
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Beyond this size GPU tends to run out of memory and total time taken increases a lot
        self.img_size = 512 if torch.cuda.is_available() else 128

        # Hides away a couple of blocks of code
        self._add_composers()

        # Init model and load it's dict from the model path
        self.cnn = vgg19.VGG('avg')
        self.cnn.load_state_dict(torch.load(Stylizer.MODEL_PATH))

        # Freeze model
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Cast model to device
        self.cnn.to(self.device)

        # Relevant layers, so far these seem to perform best
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.content_layers = ['r42']

        self.loss_layers = self.style_layers + self.content_layers

        # Init loss functions
        loss_fns_1 = [networks.GramMSELoss().to(self.device)] * \
            len(self.style_layers)
        loss_fns_2 = [nn.MSELoss().to(self.device)] * len(self.content_layers)
        self.loss_fns = loss_fns_1 + loss_fns_2

        # Init weights
        style_weights = [style_weight / n**2 for n in [64, 128, 256, 512, 512]]
        content_weights = [content_weight]
        self.weights = style_weights + content_weights

    # Conducts style transfer using the provided paths for images

    def run_standard(self, content_path, style_path, save_path, iterations, log_every=50):
        print(f'Loading content image \"{content_path}\"')
        content_img = self.load_image(content_path)

        print(f'Loading style image \"{style_path}\"')
        style_img = self.load_image(style_path)

        opt_img = Variable(content_img.clone(), requires_grad=True)
        # Uncomment below to use random noise as starting image, not advisable though
        # opt_img = Variable(torch.randn(content_img.data.size(), device=self.device), requires_grad=True)

        # Select targets for calculation
        style_targets = [networks.GramMatrix()(A).detach()
                         for A in self.cnn(style_img, self.style_layers)]
        content_targets = [A.detach()
                           for A in self.cnn(content_img, self.content_layers)]
        targets = style_targets + content_targets

        optimizer = self.optimizer([opt_img])
        iteration_count = [0]

        start = time.time()

        print("Starting style transfer")

        while iteration_count[0] <= iterations:

            def closure():
                optimizer.zero_grad()

                out = self.cnn(opt_img, self.loss_layers)
                layer_losses = [self.weights[a] * self.loss_fns[a]
                                (A, targets[a]) for a, A in enumerate(out)]

                loss = sum(layer_losses)
                loss.backward()

                iteration_count[0] += 1

                if iteration_count[0] % log_every == (log_every - 1):
                    print(
                        f'Iteration: {iteration_count[0] + 1}, loss: {loss.data}')

                return loss

            optimizer.step(closure)

        end = time.time() - start
        print(f"Time taken: {end}")

        out_img = self.post_process(opt_img.data.cpu().squeeze())
        out_img.save(save_path)
        print(f'Saved result image as \"{save_path}\"')

        return end

    def kill_yourself(self):
        self.flag_kill = True

    # Loads necessary data for run_50 calls
    def setup_run(self, content_path, style_path, save_path, progress_callback, finish_callback):
        self.progress_callback = progress_callback
        self.finish_callback = finish_callback
        self.flag_kill = False

        print(f'Loading content image \"{content_path}\"')
        self.content_img = self.load_image(content_path)

        print(f'Loading style image \"{style_path}\"')
        self.style_img = self.load_image(style_path)

        self.opt_img = Variable(self.content_img.clone(), requires_grad=True)

        style_targets = [networks.GramMatrix()(A).detach()
                         for A in self.cnn(self.style_img, self.style_layers)]
        content_targets = [A.detach() for A in self.cnn(
            self.content_img, self.content_layers)]
        self.targets = style_targets + content_targets

        self.opt = self.optimizer([self.opt_img])
        self.save_path = save_path
        self.content_path = content_path
        self.iteration = 0

        self.Worker(self).start()

    class Worker(threading.Thread):
        def __init__(self, master):
            threading.Thread.__init__(self)
            self.master = master

        # Mainly the same as run, but runs only 50 iterations at a time, for use with GUI
        def run(self):
            iterations = 50
            self.master.iteration += iterations
            iteration_count = [0]
            print('running')
            while iteration_count[0] <= iterations:

                def closure():
                    self.master.opt.zero_grad()

                    out = self.master.cnn(self.master.opt_img, self.master.loss_layers)
                    layer_losses = [self.master.weights[a] * self.master.loss_fns[a]
                                    (A, self.master.targets[a]) for a, A in enumerate(out)]

                    loss = sum(layer_losses)
                    loss.backward()

                    iteration_count[0] += 1
                    self.master.progress_callback(int(iteration_count[0]*(5/3)))
                    return loss

                if self.master.flag_kill:
                    self.master.flag_kill = False
                    messagebox.showinfo("", "Worker safely stopped")
                    break

                self.master.opt.step(closure)


            out_img = self.master.post_process(self.master.opt_img.data.cpu().squeeze())

            # name = self.content_path.split('/')[-1].split('.')[0]
            # save_p = self.save_path + "/" + name + ".jpg"
            # out_img.save(save_p)
            # print(f'Saved run_50 result image as \"{save_p}\"')
            if self.master.finish_callback(out_img):
                print('rerunning')
                self.run()
            print('quitting')

    # Loads image from path and does initial preparation
    def load_image(self, img_path):
        img = Image.open(img_path)
        img = self.prepare(img)
        img = Variable(img.unsqueeze(0).to(self.device))
        return img

    # Post-processing of tensor,
    # Does first and second post process transform and clips results to [0,1]
    def post_process(self, tensor):
        t = self.post_process_first(tensor)
        t[t > 1] = 1
        t[t < 0] = 0
        img = self.post_process_second(t)
        return img

    # Adds compositions for image processing
    def _add_composers(self):
        self.prepare = transforms.Compose([transforms.Resize(self.img_size),  # transforms.Resize(size=(self.img_size, self.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Lambda(
            lambda x: x[torch.LongTensor([2, 1, 0])]),
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                 std=[1, 1, 1]),
            transforms.Lambda(
            lambda x: x.mul_(255)),
        ])

        self.post_process_first = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                                      transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                                           std=[1, 1, 1]),
                                                      transforms.Lambda(
            lambda x: x[torch.LongTensor([2, 1, 0])]),
        ])

        self.post_process_second = transforms.Compose(
            [transforms.ToPILImage()])
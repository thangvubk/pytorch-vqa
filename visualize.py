from __future__ import division, print_function
import sys
import os.path
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse
import config
import data
from models import *
import utils
import cv2
import numpy as np

args={}
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--result', type=str, default='results/images')
args = parser.parse_args()



def get_grad_cam(pred, output):
    one_hot_pred = torch.zeros(output.size())
    one_hot_pred[pred] = 1
    output.backward(gradient=one_hot_pred)

class GradCAM(object):
    def __init__(self, model):
        self.model = model
        self.fmaps = None
        self.grads = None

    def _one_hot(self, pred, shape):
        one_hot = torch.zeros(shape) #shape = 1 x output_size
        one_hot[0][pred] = 1
        return one_hot

    def forward(self, image, question, question_len):
        # feature map equal input image
        self.fmaps = image

        output = self.model(image,  question, question_len)
        _, pred = torch.max(output.data, 1)
        return output, pred[0]

    def backward(self, output, pred):
        one_hot = self._one_hot(pred, output.size()).cuda()
        self.model.zero_grad()
        output.backward(gradient=one_hot)

        # grads equals gradient of input
        self.grads = self.fmaps.grad

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self):
        weights = self._compute_grad_weights(self.grads)

        gcam = (self.fmaps[0].data * weights[0].data).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.cpu().numpy()

def get_image_from_id(image_id):
    path = './data/images/test2015/COCO_test2015_%012d.jpg' %image_id
    image = cv2.imread(path)
    size = int(config.image_size / config.central_fraction)
    image = cv2.resize(image, (size, size))

    # center crop
    crop_start = int(size/2 - config.image_size/2)
    image = image[crop_start: crop_start + config.image_size, 
                  crop_start: crop_start + config.image_size] 
    return image

def get_question(dataset, index):
    question = dataset.questions_str[index]
    return '-'.join(question)

def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

def save_image(filename, image):
    cv2.imwrite(filename, image)

def main():

    # Dataset
    print('Creating dataset...')
    train_loader = data.get_loader(train=True) #use train loader to load vocab size
    test_loader = data.get_loader(test=True, batch_size=args.batch_size)

    # Model
    checkpoint = os.path.join(args.checkpoint)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    model_path = os.path.join(checkpoint, 'best_model.pt')
    print('Loading model...')
    model = SAN(train_loader.dataset.num_tokens)
    
    model.load_state_dict(torch.load(model_path))

    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    model.cuda()
    cudnn.benchmark = True
    model.eval()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # grad cam
    grad_cam = GradCAM(model)

    for i, (images, questions, quest_id, image_id, idx, quest_len) in enumerate(test_loader):
        images = Variable(images.cuda(), requires_grad=True) # add grad to get grad cam
        questions = Variable(questions.cuda())
        quest_len = Variable(quest_len.cuda())
        
        output, pred = grad_cam.forward(images, questions, quest_len)
        grad_cam.backward(output, pred)
        grad_map = grad_cam.generate()

        raw_image = get_image_from_id(image_id)
        question = get_question(test_loader.dataset, idx[0])
        answer = test_loader.dataset._index_to_answer(pred)
        file_name = '{}_{}_{}.jpg'.format(i, question, answer)
        file_name = os.path.join(args.result, file_name)
        save_gradcam(file_name, grad_map, raw_image)
        file_name = '{}.jpg'.format(i)
        file_name = os.path.join(args.result, file_name)
        save_image(file_name, raw_image)
        
        # take 50 samples
        if i == 200:
            break

if __name__ == '__main__':
    main()

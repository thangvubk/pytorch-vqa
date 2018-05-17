from __future__ import division, print_function
import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import progressbar
import config
import data
from models import *
import utils

args={}
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--batch-size', type=int, default=128)
args = parser.parse_args()
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
    model = SAA(train_loader.dataset.num_tokens)
    
    model.load_state_dict(torch.load(model_path))

    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True
    model.eval()

    acc = 0
    num_batches = len(test_loader.dataset)//args.batch_size
    bar = progressbar.ProgressBar(max_value=num_batches)
    running_loss = 0
    result = []
    for i, (images, questions, quest_id, _, _, quest_len) in enumerate(test_loader):
        images = Variable(images.cuda())
        questions = Variable(questions.cuda())
        quest_len = Variable(quest_len.cuda())

        outputs = model(images, questions, quest_len)
        _, preds = torch.max(outputs.data, 1)
        for j in range(preds.size(0)):
            ans_idx = preds.cpu()[j]
            answer = test_loader.dataset._index_to_answer(ans_idx)
            result.append({u'answer': answer, u'question_id': quest_id[j]})
        #acc += utils.batch_accuracy(outputs.data, labels.data).sum()
        bar.update(i, force=True)
    result = list(result)
    json.dump(result, open('result.json','w'))


if __name__ == '__main__':
    main()

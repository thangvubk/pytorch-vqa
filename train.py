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
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30, 40],
                     help='Decrease learning rate at these epochs.')
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--model', type=str, default='isan')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-epochs', type=int, default=50)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--test-only', dest='test_only', action='store_true')
args = parser.parse_args()
def main():

    # Dataset
    print('Creating dataset...')
    train_loader = data.get_loader(train=True, batch_size=args.batch_size)
    val_loader = data.get_loader(val=True, batch_size=args.batch_size)

    # Model
    checkpoint = os.path.join(args.checkpoint)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    model_path = os.path.join(checkpoint, 'best_model.pt')
    print('Loading model...')
    print(args.model)
    model = get_vqa_model(args.model, train_loader.dataset.num_tokens)
    #model = resnet50_CA()
    #print(model)
    # Test only

    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    # optim
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.5)
    log_softmax = nn.LogSoftmax().cuda()
    
    # Log 
    log_file = os.path.join(checkpoint, 'log.json')
    writer = SummaryWriter(checkpoint)
    best_val_acc = -1
    # Train and val
    for epoch in range(args.num_epochs):
        # Train
        scheduler.step()
        learning_rate = optimizer.param_groups[0]['lr']
        print('Start training epoch {}. Learning rate {}'.format(epoch, learning_rate))
        writer.add_scalar('Learning rate', learning_rate, epoch)
        model.train()
        num_batches = len(train_loader.dataset)//args.batch_size
        bar = progressbar.ProgressBar(max_value=num_batches)
        running_loss = 0
        train_acc = 0
        for i, (images, questions, labels, idx, quest_len) in enumerate(train_loader):
            images = Variable(images.cuda())
            questions = Variable(questions.cuda())
            labels = Variable(labels.cuda())
            quest_len = Variable(quest_len.cuda())

            
            optimizer.zero_grad()
            outputs = model(images, questions, quest_len)
            nll = -log_softmax(outputs)
            loss = (nll * labels / 10).sum(dim=1).mean()

            train_acc += utils.batch_accuracy(outputs.data, labels.data).sum()
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            bar.update(i, force=True)
            writer.add_scalar('Training instance loss', loss.data[0], epoch*num_batches + i)
        train_acc = train_acc/len(train_loader.dataset)*100
        train_loss = running_loss/num_batches
        print('Training loss %f' %train_loss)
        print('Training acc %.2f' %train_acc)
        writer.add_scalar('Training loss', train_loss, epoch)

        # Validate
        model.eval()
        val_acc = 0
        num_batches = len(val_loader.dataset)//args.batch_size
        bar = progressbar.ProgressBar(max_value=num_batches)
        running_loss = 0
        for i, (images, questions, labels, idx, quest_len) in enumerate(val_loader):
            images = Variable(images.cuda())
            questions = Variable(questions.cuda())
            labels = Variable(labels.cuda())
            quest_len = Variable(quest_len.cuda())

            outputs = model(images, questions, quest_len)
            nll = -log_softmax(outputs)

            loss = (nll * labels / 10).sum(dim=1).mean()
            val_acc += utils.batch_accuracy(outputs.data, labels.data).sum()
            running_loss += loss.data[0]
            bar.update(i, force=True)

        val_acc = val_acc/len(val_loader.dataset)*100
        val_loss = running_loss/num_batches
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        print('Validation loss %f' %(running_loss/num_batches))
        print('Validation acc', val_acc)
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation acc', val_acc, epoch)
        print()
    print('Best validation acc %.2f' %best_val_acc)
    writer.export_scalars_to_json(log_file)
    writer.close()

if __name__ == '__main__':
    main()

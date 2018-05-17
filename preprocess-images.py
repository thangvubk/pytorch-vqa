import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import config
import data
import utils
#from resnet import resnet as caffe_resnet
from models import *

MODEL = 'resnet' # (vgg or resnet) select model to extract image features

class TruncatedResNet(nn.Module):
    def __init__(self):
        super(TruncatedResNet, self).__init__()
        self.model = resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

class TruncatedVgg(nn.Module):
    def __init__(self):
        super(TruncatedVgg, self).__init__()
        self.model = vgg19(pretrained=True)

    def forward(self, x):
        return self.model.features(x)

def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    cudnn.benchmark = True
    
    if MODEL is 'resnet':
        net = TruncatedResNet()
    elif MODEL is 'vgg':
        net = TruncatedVgg()
    else:
        raise Exception('Unknown model', MODEL)

    net.cuda()
    net.eval()

    loader = create_coco_loader(config.train_path, config.val_path, config.test_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            imgs = Variable(imgs.cuda(async=True), volatile=True)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()

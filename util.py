from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os.path as osp
import os

from sklearn.metrics import f1_score

#plot tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# t-SNE packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def f1_loss(y_pred, y_true, is_training=False):
    '''
    Calculate F1 score. Can work with gpu tensors    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    # print(y_true)
    # print(y_pred)

    y_true = y_true.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()

    F1 = f1_score(y_true, y_pred, average='macro')

    return F1 * 100

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def load_model(ckpt_path, model, optimizer):
    print('==> Loading...')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

'''
Section to plot the t-SNE representation - Generated per epoch
'''
def generate_tsne_from_feat_embedding(featureList, labelList, num_samples = 1000, use_all=True):
    # Check if the total no of features = no of labels found
    assert len(featureList) == len(labelList)
    assert len(featureList) > 0
    feature = featureList[0]
    label = labelList[0]
    for i in range(1, len(labelList)):
        feature = torch.cat([feature,featureList[i]],dim=0)
        label = torch.cat([label, labelList[i]], dim=0)

    feat = []
    lbl = []
    # Filter out the class wise feature set
    for i in range(torch.unique(label).shape[0]):
        mask = label == i
        lbel = i * torch.ones(feature[mask,:].shape[0])
        if use_all:
            num_samples = lbel.shape[0]
        if i == 0:
            feat = feature[mask,:][:num_samples, :]
            lbl = lbel[:num_samples]
        else:
            feat = torch.cat([feat,feature[mask,:][:num_samples, :]])      
            lbl = torch.cat((lbl, lbel[:num_samples]), dim=0)
        
    # create scaler
    scaler = MinMaxScaler()
    # Move to CPU - TODO : Need to fix this going forward 
    feature =scaler.fit_transform(feat.cpu().detach().numpy())
    label = lbl.cpu().detach().numpy()
    # Using PCA to reduce dimension to a reasonable dimension as recommended in
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    feature = PCA(n_components=0.99).fit_transform(feature) # earlier was 50
    feature_embedded = TSNE(n_components=2, 
                            learning_rate='auto', 
                            init='random', 
                            perplexity=50).fit_transform(feature)
    return feature_embedded, label

def plot_tsne(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    # TODO : Make it generic for all datasets
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join('output', prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


class scale_255(object):
    def __call__(self, img):
        return img*255.
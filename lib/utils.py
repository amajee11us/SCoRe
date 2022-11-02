from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import shutil
import os
import os.path as osp
import logging

#plot tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# t-SNE packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def Logger(cfg):
    # clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #Make output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # set up logging to file
    logging.basicConfig(
        filename=os.path.join(cfg.OUTPUT_DIR, 'log.log'),
        filemode='a',
        level=logging.INFO,
        format=
        '[%(asctime)s]{%(filename)s:%(lineno)d}%(levelname)s- %(message)s',
        datefmt='%H:%M:%S')
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    return logger


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 (say) epochs"""
    lr = cfg.TRAIN.LEARNING_RATE * (0.1**(epoch // cfg.TRAIN.LR_DECAY_STEP))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(cfg, state, is_best):
    file_path = cfg.OUTPUT_DIR
    filename = os.path.join(file_path, "model_" + str(state['epoch']) + ".pth")
    torch.save(state, filename)
    if is_best:
        # just create a copy
        shutil.copyfile(filename, os.path.join(file_path, "model_best.pth"))


def get_output_tb_dir(cfg):
    """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(osp.join(cfg.OUTPUT_DIR, 'tensorboard'))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_ckpt_dir(cfg):
    # Define the output path
    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ARCH + "_" + cfg.EXP_NAME)

    if os.path.exists(out_dir):
        logging.info(
            "Path already exists, would be appending to the same path.")
    else:
        try:
            os.makedirs(out_dir)
            logging.info("Created working directory at : {}".format(out_dir))
        except Exception as e:
            logging.error("Error while directory creation: {}".format(e))

    return out_dir


def get_target_device(cfg, gpu_num=0):
    '''
    Returns a device to dump all workings into.
    '''
    device = torch.device("cpu")
    if "gpu" in cfg.DEVICE:
        if not torch.cuda.is_available():
            logging.warning(
                f'CUDA is NOT available. Fall-back initiated to CPU.')
            # fallback already initialized so no need to do it again
        else:
            # Here we have CUDA
            device = torch.device("cuda:" + str(gpu_num))

    return device

'''
Section to plot the t-SNE representation - Generated per epoch
'''
def generate_tsne_from_feat_embedding(featureList, labelList):
    # Check if the total no of features = no of labels found
    assert len(featureList) == len(labelList)
    assert len(featureList) > 0
    feature = featureList[0]
    label = labelList[0]
    for i in range(1, len(labelList)):
        feature = torch.cat([feature,featureList[i]],dim=0)
        label = torch.cat([label, labelList[i]], dim=0)

    # Move to CPU - TODO : Need to fix this going forward 
    feature =feature.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    # Using PCA to reduce dimension to a reasonable dimension as recommended in
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    feature = PCA(n_components=50).fit_transform(feature)
    feature_embedded = TSNE(n_components=2).fit_transform(feature)
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

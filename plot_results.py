import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import argparse
from models.lstm import LSTMClassificationModel
from models.mmvtt import MMVTT
from models.resnet import ResnetClassificationModel
from datasets.dataset import MMVTDataset, split_data
from omegaconf import OmegaConf
import torch

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_results(args):

    """ parse files """
    dir = args.path
    files = os.listdir(dir)
    for file in files:
        if 'events.out' in file:
            file_log = file
        elif '.pth' in file:
            file_wts = file
        elif '.yaml' in file:
            file_conf = file
    
    """ curves """
    ea = event_accumulator.EventAccumulator(os.path.join(dir, file_log)) 
    ea.Reload()
    val_loss = ea.scalars.Items('val/loss')
    val_accuracy = ea.scalars.Items('val/accuracy')
    val_recall = ea.scalars.Items('val/recall')
    # plot loss
    steps = [i.step for i in val_loss]
    loss = [i.value for i in val_loss]
    plt.figure()
    plt.plot(steps, loss)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(dir, 'loss.png'))
    # plot accuracy
    steps = [i.step for i in val_accuracy]
    acc = [i.value for i in val_accuracy]
    plt.figure()
    plt.plot(steps, acc)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(dir, 'accuracy.png'))
    # plot recall
    steps = [i.step for i in val_recall]
    recall = [i.value for i in val_recall]
    plt.figure()
    plt.plot(steps, recall)
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(os.path.join(dir, 'recall.png'))

    """ confusion matrix """
    cfg = OmegaConf.load(os.path.join(dir, file_conf))
    print("using {} device.".format(cfg.device))
    set_seed(cfg.random_seed)

    """ load and split dataset """
    vt_dataset = MMVTDataset(
        data_dir=cfg.data_dir,
        visual_mod_trans_mapping=cfg.visual_modality,
        random_visual=cfg.random_visual,
        tactile_mod_trans_mapping=cfg.tactile_modality,
        size=cfg.img_size,
    )
    cfg.num_classes = len(vt_dataset.labels)
    train_data, val_data = split_data(vt_dataset, cfg)

    """ load model """
    if len(cfg.visual_modality)> 0:
        visual_type = 1 if cfg.random_visual else len(list(cfg.visual_modality.keys()))
    else:
        visual_type= 0

    tactile_type = len(list(cfg.tactile_modality.keys()))
    if cfg.alg == 'MMVTT':
        """ transformer model"""
        net = MMVTT(
            types=[visual_type, tactile_type],
            **cfg,
        )
    elif cfg.alg == 'resnet':
        """ resnet model """
        net = ResnetClassificationModel(args=cfg, types=tactile_type+visual_type)
    elif cfg.alg == 'lstm':
        """ LSTM model """
        net = LSTMClassificationModel(args=cfg)
    else:
        raise NotImplementedError(f'{cfg.alg} is not implemented!!!')
    net.load_state_dict(torch.load(os.path.join(dir, file_wts)))
    net.to(cfg.device)
    net.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for data in val_data:
            visual, tactile, label = data['visual'], data['tactile'], data['label']
            visual, tactile, label = visual.to(cfg.device), tactile.to(cfg.device), label.to(cfg.device)
            _, pred = net(visual, tactile)
            pred_labels = np.argmax(pred.cpu().data.numpy(), axis=1)
            true_labels = np.argmax(label.cpu().data.numpy(),axis=1)
            preds.append(pred_labels)
            gts.append(true_labels)
        preds = np.concatenate(preds, 0).reshape(-1,)
        gts = np.concatenate(gts, 0).reshape(-1,)
    # print(preds, gts)
    nums = np.array([np.sum(gts==i) for i in range(12)]).reshape(12,1).astype(np.float32)
    confusion_mat = confusion_matrix(gts, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat/nums, display_labels=[1,2,3,4,5,6,7,8,9,10,11,12])
    disp.plot(
        include_values=True,            
        cmap=plt.cm.Blues,                 
        ax=None,                        
        xticks_rotation="horizontal",   
        values_format=".0%",
    )
    # label_font = {'size':'18'}  # Adjust to fit
    disp.ax_.set_xlabel('Predicted labels')
    disp.ax_.set_ylabel('True labels')
    disp.ax_.set_title('Confusion Matrix (%)')
    plt.savefig(os.path.join(dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path of log directory')
    args = parser.parse_args()
    plot_results(args)
import numpy as np
import argparse
import copy, time
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import time
from sklearn.metrics import recall_score

from models.mmvtt import MMVTT
from models.resnet import ResnetClassificationModel
from models.lstm import LSTMClassificationModel
from datasets.dataset import MMVTDataset, split_data

def add_noise(batch_visual, amount=0.01, s_vs_p=0.5):
    if batch_visual.ndim != 4:
        return batch_visual
    B, C, H, W = batch_visual.shape
    num_salt = np.ceil(amount * H*W * s_vs_p)
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in [H,W,C]]
    batch_visual[..., coords[0],coords[1]] = (torch.ones((B, C, 1)) - torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1)) / torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1)
    num_pepper = np.ceil(amount * H*W  * (1. - s_vs_p))
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in [H,W]]
    batch_visual[..., coords[0],coords[1]] = (torch.zeros((B, C, 1)) - torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1)) / torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1)
    return batch_visual

def trainer(net, loss, optimizer, scheduler, dataset, writer, log_path, args):
    used_modalities = list(args.tactile_modality.keys())
    if 'v' in used_modalities:
        tac_v_index = used_modalities.index('v')
    elif 'vt' in used_modalities: 
        tac_v_index = used_modalities.index('vt')
    else:
        tac_v_index = -1
    print(tac_v_index)
    num_epoch = args.epoch
    train_data_length = dataset["train"].__len__() * args.batch_size
    val_data_length = dataset["val"].__len__()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_acc = 0.0

    net = net.to(args.device)

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        net.train()  
        for data in tqdm(dataset["train"]):
            visual, tactile, label = data['visual'], data['tactile'], data['label']
            if tac_v_index >= 0 :
                tactile[:, tac_v_index] = add_noise(tactile[:,tac_v_index], amount=0.01) 
            visual, tactile, label = visual.to(args.device), tactile.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            _, train_pred = net(visual, tactile) 
            batch_loss = loss(train_pred, label.float())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == np.argmax(label.cpu().data.numpy(),axis=1))
            train_loss += batch_loss.item()
        
        writer.add_scalar('train/loss', train_loss / train_data_length, epoch)
        writer.add_scalar('train/accuracy', train_acc / train_data_length, epoch)

        if epoch % args.val_interval == 0:
            y_preds = []
            y_trues = []
            net.eval()
            with torch.no_grad():
                for data in tqdm(dataset["val"]):
                    visual, tactile, label = data['visual'], data['tactile'], data['label']
                    if tac_v_index >= 0 :
                        tactile[:, tac_v_index] = add_noise(tactile[:,tac_v_index], amount=0.02) 
                    visual, tactile, label = visual.to(args.device), tactile.to(args.device), label.to(args.device)
                    _, val_pred = net(visual, tactile)
                    batch_loss = loss(val_pred, label.float())

                    pred_labels = np.argmax(val_pred.cpu().data.numpy(), axis=1)
                    true_labels = np.argmax(label.cpu().data.numpy(),axis=1)
                    val_acc += np.sum(pred_labels == true_labels)
                    val_loss += batch_loss.item()
                    y_preds.append(pred_labels)
                    y_trues.append(true_labels)

                y_preds = np.concatenate(y_preds)
                y_trues = np.concatenate(y_trues)
                val_recall = recall_score(y_pred=y_preds, y_true=y_trues, average='macro')
                print('\n')
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                    (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                    train_acc / train_data_length, train_loss / train_data_length, val_acc / val_data_length,
                    val_loss / val_data_length))
                
                
            writer.add_scalar('val/loss', val_loss / val_data_length, epoch)
            writer.add_scalar('val/accuracy', val_acc / val_data_length, epoch)
            writer.add_scalar('val/recall', val_recall, epoch)

            writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

            if 1.5 * val_acc + train_acc > best_acc:
                best_acc = 1.5 * val_acc + train_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                best_train_acc = train_acc
                best_val_acc = val_acc

            writer.add_scalar('best/train_acc', best_train_acc / train_data_length, epoch)
            writer.add_scalar('best/val_acc', best_val_acc / val_data_length, epoch)
            # torch.save(net.state_dict(),f'{log_path}/{epoch}.pth')
        scheduler.step()

    best_train_acc = best_train_acc / train_data_length
    best_val_acc = best_val_acc / val_data_length
    torch.save(best_model_wts, "{0}/val_acc{1:.2f}-train_acc{2:.2f}.pth".format(log_path, best_val_acc, best_train_acc))

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--cfg', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def main(args):
    """ Load config and set up for training experiment """
    cfg = OmegaConf.load(args.cfg)
    print("using {} device.".format(cfg.device))
    set_seed(cfg.random_seed)
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(cfg.exp, cur_time)
    os.makedirs(log_path)
    OmegaConf.save(cfg, os.path.join(log_path, 'config.yaml'))
    writer = SummaryWriter(log_path)

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
    dataloader_dict = {'train': train_data, 'val': val_data}

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

    """ loss function """
    if cfg.loss == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{cfg.optimizer} not implemented!!!')
    
    """ optimizer """
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    else:
        raise NotImplementedError(f'{cfg.optimizer} not implemented!!!')
    
    """ scheduler """

    if cfg.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.period, gamma=cfg.gamma)
    else: 
        NotImplementedError(f'{cfg.scheduler} not implemented!!!')
    
    """ start train """
    trainer(net, loss, optimizer, scheduler, dataloader_dict, writer, log_path, cfg)

if __name__ == "__main__":
    args = parse_args()
    main(args)

    
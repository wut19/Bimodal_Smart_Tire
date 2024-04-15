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

from models.vtt import VTT
from datasets.dataset import VTDataset, split_data

def trainer(net, loss, optimizer, dataset, writer, log_path, args):
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
        
        net.eval()
        with torch.no_grad():
            for data in tqdm(dataset["val"]):
                visual, tactile, label = data['visual'], data['tactile'], data['label']
                visual, tactile, label = visual.to(args.device), tactile.to(args.device), label.to(args.device)
                _, val_pred = net(visual, tactile)
                batch_loss = loss(val_pred, label.float())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == np.argmax(label.cpu().data.numpy(),axis=1))
                val_loss += batch_loss.item()
            print('\n')
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                   train_acc / train_data_length, train_loss / train_data_length, val_acc / val_data_length,
                   val_loss / val_data_length))
            
        writer.add_scalar('val/loss', val_loss / val_data_length, epoch)
        writer.add_scalar('val/accuracy', val_acc / val_data_length, epoch)

        if 1.5 * val_acc + train_acc > best_acc:
            best_acc = 1.5 * val_acc + train_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            best_train_acc = train_acc
            best_val_acc = val_acc

        writer.add_scalar('best/train_acc', best_train_acc / train_data_length, epoch)
        writer.add_scalar('best/val_acc', best_val_acc / val_data_length, epoch)

    best_train_acc = best_train_acc / train_data_length
    best_val_acc = best_val_acc / val_data_length
    torch.save(best_model_wts, f"{log_path}/val_acc{0:.3f}-train_acc{1:.3f}.pth".format(best_val_acc, best_train_acc))

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--cfg', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    """ Load config and set up for training experiment """
    cfg = OmegaConf.load(args.cfg)
    print("using {} device.".format(cfg.device))
    set_seed(cfg.random_seed)
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(cfg.exp, cur_time)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    """ load and split dataset """
    vt_dataset = VTDataset(
        data_dir=cfg.data_dir,
        used_visual_modalities=cfg.visual_modality,
        random_visual=cfg.random_visual,
        use_tactile=cfg.tactile_modality,
        size=cfg.img_size,
        crop_size=cfg.crop_size,
        )
    cfg.num_classes = len(vt_dataset.labels)
    train_data, val_data = split_data(vt_dataset, cfg)
    dataloader_dict = {'train': train_data, 'val': val_data}

    """ load model """
    if len(cfg.visual_modality)> 0:
        visual_type = 1 if cfg.random_visual else len(cfg.visual_modality)
    else:
        visual_type= 0
    net = VTT(
        visual_size=cfg.img_size,
        visual_patch_size=cfg.patch_size,
        visual_type= visual_type,
        tactile_size=cfg.img_size,
        tactile_patch_size=cfg.patch_size,
        use_tactile=cfg.tactile_modality,
        **cfg,
    )
    
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
    trainer(net, loss, optimizer, dataloader_dict, writer, log_path, cfg)

if __name__ == "__main__":
    args = parse_args()
    main(args)

    
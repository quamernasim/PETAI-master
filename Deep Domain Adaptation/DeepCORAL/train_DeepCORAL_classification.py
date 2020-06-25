import torch
import numpy as np
import argparse
import itertools
import os
from os.path import join as pjoin
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from core.models import model_classification as DeepCORAL_model
import core.loss
import torchvision.utils as vutils
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
from core import data_loader_netherlands, data_loader_canada
from core.metrics import runningScore, save_classification_csv
from core.utils import np_to_tb,AverageMeter, inter_and_union
from PIL import Image
from torchvision import transforms 
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
from core import utils
from torch.utils import data

# Fix the random seeds: 
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)

def normalize(data, _max, _min):
    return ((data - _min)/(_max - _min))

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_fname = pjoin(val_fname, 'best_checkpoint.pth')

    log_dir = os.path.join(val_fname, "Tensorboard_Records")
    writer = SummaryWriter(log_dir=log_dir)
    
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->
    # Setup Augmentations
    #if args.aug:
    #    data_aug = Compose(
    #        [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
    #else:
    #    data_aug = None
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->

    source_train_set = data_loader_netherlands.PatchLoader(is_transform=True, split='train', augmentations=None)
    source_sampler = source_train_set.get_sampler()
    
    target_train_set = data_loader_canada.PatchLoader(is_transform=True, split='train', augmentations=None)
    target_sampler = target_train_set.get_sampler()

    # Without Augmentation:
    source_val_set = data_loader_netherlands.PatchLoader(is_transform=True, split='valid')
                                                          
    target_val_set = data_loader_canada.PatchLoader(is_transform=True, split='valid')

    n_classes = source_train_set.n_classes

    source_trainloader = data.DataLoader(source_train_set,
                                        batch_size=args.netherlands_batch_size,
                                        num_workers=1,
                                        sampler=source_sampler)
                                        
    target_trainloader = data.DataLoader(target_train_set,
                                        batch_size=args.canada_batch_size,
                                        num_workers=1,
                                        sampler=target_sampler)
                                  
    source_valloader = data.DataLoader(source_val_set,
                                        batch_size=args.netherlands_batch_size,
                                        num_workers=1)
                                        
    target_valloader = data.DataLoader(target_val_set,
                                        batch_size=args.canada_batch_size,
                                        num_workers=1)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
    source_running_metrics_val = runningScore(n_classes)
    target_running_metrics_val = runningScore(n_classes)

    # Setup Model edited by Tannistha
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = DeepCORAL_model.DeepCORAL(num_classes = n_classes)
            checkpoint = torch.load(args.resume)
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
            model.train()
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        model = DeepCORAL_model.DeepCORAL(num_classes = n_classes)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Use as many GPUs as we can
    model = model.to(device)  # Send to GPU

    # PYTROCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU,

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        if args.optim in ["sgd", "SGD", "Sgd"]:
            optimizer = torch.optim.SGD(model.parameters(),lr=args.base_lr, weight_decay=0.0001, momentum=0.9)
        elif args.optim in ["adam", "ADAM", "Adam"]:
            optimizer = torch.optim.Adam(model.parameters(),lr=args.base_lr, weight_decay=0.0001, amsgrad=True)
        elif args.optim in ["adadelta", "ADADELTA", "AdaDelta", "Adadelta"]:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=args.base_lr, rho=0.9, eps=1e-06, weight_decay=0.0001)
        else:
            print("Unknown Optimizer! Choose from [sgd, adam, adadelta]")
        
        
    if args.train:
        loss_fn = F.cross_entropy
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        
        start_epoch = 0

    if args.class_weights:
        # weights are inversely proportional to the frequency of the classes in the training set
        class_weights = torch.tensor(
            [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
    else:
        class_weights = None

    best_mca = -100.0
    
#<----------------------------------------------------------------CHANGE ACCORDINGLY------------------------------------------------------------------------------->
    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
#<----------------------------------------------------------------CHANGE ACCORDINGLY------------------------------------------------------------------------------->
    
    for arg in vars(args):
        text = arg + ': ' + str(getattr(args, arg))
        writer.add_text('Parameters/', text)
    
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        sum_loss_train, classification_loss_train, coral_loss_train, total_iteration = 0, 0, 0, 0
        
        train_steps = min(len(source_trainloader), len(target_trainloader))
        
        for i, ((source_image, source_label), (target_image, _)) in enumerate(zip(source_trainloader, target_trainloader)):
    
            source_image_original, source_labels_original = source_image, source_label
            target_image_original = target_image
            
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image = target_image.to(device)
            
            optimizer.zero_grad()
            
            if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                source_output, target_output = model(source_image, target_image)
            elif args.backbone in ["SeismicNet", "seismicnet"]:
                source_output, target_output = model(source_image, target_image)
            else:
                print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
            
            pred = source_output.detach().max(1)[1].cpu().numpy()
            gt = source_label.detach().cpu().numpy()
            
            running_metrics.update(gt, pred)

            classification_loss = loss_fn(input=source_output, target=torch.squeeze(source_label), weight=class_weights)
            
            if args.normalize_loss:
                if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                    classification_loss = normalize(classification_loss, 1.5045, 0)
                elif args.backbone in ["SeismicNet", "seismicnet"]:
                    classification_loss = normalize(classification_loss, 1.6350, 0)
                else:
                    print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
                    
            #lambda is weightening factor for CORAL Loss to match the scale for classification loss
            
            if ("_".join(args.__lambda.split("_")[:-1])) == "epoch_based":
                _lambda = ((epoch+1)/args.n_epoch) * int(args.__lambda.split("_")[-1]) #As given in Repo
            elif args.__lambda.split("_")[0] in ["Constant", "constant"]:
                _lambda = int(args.__lambda.split("_")[1]) #Quamer, temporary
            else:
                print("unknown lambda. Enter in form [epoch_based_value, constant_value]")
            
            if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                coral_loss = DeepCORAL_model.CORAL(source_output, target_output)
            elif args.backbone in ["SeismicNet", "seismicnet"]:
                coral_loss = DeepCORAL_model.CORAL(source_output, target_output)
            else:
                print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
                
            if args.normalize_loss:
                if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                    coral_loss = normalize(coral_loss, 0.00605, 0)
                elif args.backbone in ["SeismicNet", "seismicnet"]:
                    coral_loss = normalize(coral_loss, 0.0515, 0)
                else:
                    print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
            
            coral_loss_weight = 0
            
            sum_loss = (coral_loss * _lambda) + classification_loss

            classification_loss_train += classification_loss.item()
            coral_loss_train += coral_loss.item()
            sum_loss_train += sum_loss.item()
            
            sum_loss.backward()

            if args.clip != 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            
            total_iteration = total_iteration + 1

            if (i) % 20 == 0:
                print('epoch: {0}/{1}\t\t iter: {2}/{3}\t\t LOSS( Classification Loss:{4:.4f}\t\t CORAL Loss:{5:.4f}\t\t Total Loss:{6:.4f} )'.format(
                            epoch + 1, args.n_epoch, i + 1, train_steps, classification_loss.item(), coral_loss.item(), sum_loss.item()))

#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->
            #numbers = [0]
            #if i in numbers:
            #    # number 0 image in the batch
            #    tb_original_image = vutils.make_grid(
            #        image_original[0][0], normalize=True, scale_each=True)
            #    writer.add_image('train/original_image',
            #                     tb_original_image, epoch + 1)

            #    labels_original = labels_original.numpy()[0]
            #    correct_label_decoded = train_set.decode_segmap(np.squeeze(labels_original))
            #    writer.add_image('train/original_label',np_to_tb(correct_label_decoded), epoch + 1)
            #    out = F.softmax(outputs, dim=1)

            #    # this returns the max. channel number:
            #    prediction = out.max(1)[1].cpu().numpy()[0]
            #    # this returns the confidence:
            #    confidence = out.max(1)[0].cpu().detach()[0]
            #    tb_confidence = vutils.make_grid(
            #        confidence, normalize=True, scale_each=True)

            #    decoded = train_set.decode_segmap(np.squeeze(prediction))
            #    writer.add_image('train/predicted', np_to_tb(decoded), epoch + 1)
            #    writer.add_image('train/confidence', tb_confidence, epoch + 1)

            #    unary = outputs.cpu().detach()
            #    unary_max = torch.max(unary)
            #    unary_min = torch.min(unary)
            #    unary = unary.add((-1*unary_min))
            #    unary = unary/(unary_max - unary_min)

            #    for channel in range(0, len(class_names)):
            #        decoded_channel = unary[0][channel]
            #        tb_channel = vutils.make_grid(
            #            decoded_channel, normalize=True, scale_each=True)
            #        writer.add_image(f'train_classes/_{class_names[channel]}', tb_channel, epoch + 1)
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->
        
        # Average metrics, and save in writer()
        classification_loss_train /= total_iteration
        coral_loss_train /= total_iteration
        sum_loss_train /= total_iteration
        
        score = running_metrics.get_classification_scores()
        writer.add_scalar('train/Mean Class Acc', score['Mean Class Acc: '], epoch+1)
        writer.add_scalar('train/classification_loss', classification_loss_train, epoch+1)
        writer.add_scalar('train/coral_loss', coral_loss_train, epoch+1)
        writer.add_scalar('train/sum_loss', sum_loss_train, epoch+1)
        running_metrics.reset()
        
        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval()
                
                source_classification_loss, target_classification_loss, source_classification_loss_val, target_classification_loss_val, total_iteration_source_val, total_iteration_target_val = 0, 0, 0, 0, 0, 0
                print()
                print("=====================================================================================================================================================================================")
                print()
                
                for i_val_source, (source_image_val, source_label_val) in enumerate(source_valloader):
                    
                    source_image_val, source_label_val = source_image_val.to(device), source_label_val.to(device)
                    
                    if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                        source_outputs_val, _ = model(source_image_val, source_image_val)
                    elif args.backbone in ["SeismicNet", "seismicnet"]:
                        source_outputs_val, _ = model(source_image_val, source_image_val)
                    else:
                        print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
                    
                    source_pred = source_outputs_val.detach().max(1)[1].cpu().numpy()
                    source_gt = source_label_val.detach().cpu().numpy()
                    
                    source_running_metrics_val.update(source_gt, source_pred)
                    
                    source_classification_loss = loss_fn(input=source_outputs_val, target=torch.squeeze(source_label_val), weight=class_weights)
                    
                    if args.normalize_loss:
                        if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                            source_classification_loss = normalize(source_classification_loss, 1.5045, 0)
                        elif args.backbone in ["SeismicNet", "seismicnet"]:
                            source_classification_loss = normalize(source_classification_loss, 1.6350, 0)
                        else:
                            print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")

                    source_classification_loss_val += source_classification_loss.item()
                    
                    total_iteration_source_val = total_iteration_source_val + 1
                    
                    if (i_val_source) % 20 == 0:
                        print('epoch: {0}/{1}\t\t iter: {2}/{3}\t\t Validation Classification Loss(SOURCE): {4:.4f}'.format(epoch + 1, 
                                                        args.n_epoch, i_val_source + 1, len(source_valloader), source_classification_loss.item()))
                    
                print()
                print()
                
                for i_val_target, (target_image_val, target_label_val) in enumerate(target_valloader):
                    
                    target_image_val, target_label_val = target_image_val.to(device), target_label_val.to(device)
                    
                    if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                        target_outputs_val, _ = model(target_image_val, target_image_val)
                    elif args.backbone in ["SeismicNet", "seismicnet"]:
                        target_outputs_val, _ = model(target_image_val, target_image_val)
                    else:
                        print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")
                    
                    target_pred = target_outputs_val.detach().max(1)[1].cpu().numpy()
                    target_gt = target_label_val.detach().cpu().numpy()
                    
                    target_running_metrics_val.update(target_gt, target_pred)
                    
                    target_classification_loss = loss_fn(input=target_outputs_val, target=torch.squeeze(target_label_val), weight=class_weights)
                    
                    if args.normalize_loss:
                        if args.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
                            target_classification_loss = normalize(target_classification_loss, 1.5045, 0)
                        elif args.backbone in ["SeismicNet", "seismicnet"]:
                            target_classification_loss = normalize(target_classification_loss, 1.6350, 0)
                        else:
                            print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")

                    target_classification_loss_val += target_classification_loss.item()
                    
                    total_iteration_target_val = total_iteration_target_val + 1

                    if (i_val_target) % 20 == 0:
                        print('epoch: {0}/{1}\t\t iter: {2}/{3}\t\t Validation Classification Loss(TARGET): {4:.4f}'.format(epoch + 1, 
                                                        args.n_epoch, i_val_target + 1, len(target_valloader), target_classification_loss.item()))
                                        
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->                 
                    #numbers = [0]
                    #if i_val in numbers:
                    #    # number 0 image in the batch
                    #    tb_original_image = vutils.make_grid(
                    #        image_original[0][0], normalize=True, scale_each=True)
                    #    writer.add_image('val/original_image',
                    #                     tb_original_image, epoch)
                    #    labels_original = labels_original.numpy()[0]
                    #    correct_label_decoded = train_set.decode_segmap(
                    #        np.squeeze(labels_original))
                    #    writer.add_image('val/original_label',
                    #                     np_to_tb(correct_label_decoded), epoch + 1)

                    #    out = F.softmax(outputs_val, dim=1)

                    #    # this returns the max. channel number:
                    #    prediction = out.max(1)[1].cpu().detach().numpy()[0]
                    #    # this returns the confidence:
                    #    confidence = out.max(1)[0].cpu().detach()[0]
                    #    tb_confidence = vutils.make_grid(
                    #        confidence, normalize=True, scale_each=True)

                    #    decoded = train_set.decode_segmap(
                    #        np.squeeze(prediction))
                    #    writer.add_image('val/predicted', np_to_tb(decoded), epoch + 1)
                    #    writer.add_image('val/confidence',
                    #                     tb_confidence, epoch + 1)

                    #    unary = outputs.cpu().detach()
                    #    unary_max, unary_min = torch.max(
                    #        unary), torch.min(unary)
                    #    unary = unary.add((-1*unary_min))
                    #    unary = unary/(unary_max - unary_min)

                    #    for channel in range(0, len(class_names)):
                    #        tb_channel = vutils.make_grid(
                    #            unary[0][channel], normalize=True, scale_each=True)
                    #        writer.add_image(
                    #            f'val_classes/_{class_names[channel]}', tb_channel, epoch + 1)
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->

                source_classification_loss_val /= total_iteration_source_val
                target_classification_loss_val /= total_iteration_target_val
                
                source_score = source_running_metrics_val.get_classification_scores()
                target_score = target_running_metrics_val.get_classification_scores()
                
                save_classification_csv(source_score, val_fname, epoch, "Source", n_classes)
                save_classification_csv(target_score, val_fname, epoch, "Target", n_classes)
                
                writer.add_scalar('val_Source/Mean Class Acc', source_score['Mean Class Acc: '], epoch+1)
                writer.add_scalar('val_Source/Loss', source_classification_loss_val, epoch+1)
                source_running_metrics_val.reset()
                
                writer.add_scalar('val_Target/Mean Class Acc', target_score['Mean Class Acc: '], epoch+1)
                writer.add_scalar('val_Target/Loss', target_classification_loss_val, epoch+1)
                target_running_metrics_val.reset()
                
                #Model saving based on Target Validation
                if target_score['Mean Class Acc: '] >= best_mca:
                    best_mca = target_score['Mean Class Acc: ']
                    best_epoch = epoch + 1
                    torch.save({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, model_fname)
                                
                print()
                print()
                print("Most Recent Checkpoint Saved at Epoch Number {0}.".format(best_epoch))                                
                print()
                print("=====================================================================================================================================================================================")
                print()

#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->
        #else:  # validation is turned off:
        #    # just save the latest model:
        #    if (epoch+1) % 5 == 0:
        #        model_dir = os.path.join(
        #            log_dir, f"{args.arch}_ep{epoch+1}_model.pkl")
        #       #torch.save(model, model_dir)
        #        torch.save({'epoch': epoch + 1,
        #                'state_dict': model.state_dict(),
        #                'optimizer': optimizer.state_dict(),}, model_fname % (epoch + 1))
#<--------------------------------------------------------------------------NOT NECESSARY-------------------------------------------------------------------------->
                        
        #writer.add_scalar('train/epoch_lr', optimizer.param_groups[0]["lr"], epoch+1)
        
    writer.close()
    print("Best Checkpoint Saved at epoch number {0}.".format(best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='SeismicNet',
                        help='Architecture to use [\'SeismicNet, DeConvNet, DeConvNetSkip\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=101,
                        help='# of the epochs')
    parser.add_argument('--netherlands_batch_size', nargs='?', type=int, default=64,
                        help='Batch Size of Netherlands Dataset')
    parser.add_argument('--canada_batch_size', nargs='?', type=int, default=32,
                        help='Batch Size of Canada Dataset')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--clip', nargs='?', type=float, default=0.1,
                        help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val', nargs='?', type=float, default=0.2,
                        help='percentage of the training data for validation')
    parser.add_argument('--stride', nargs='?', type=int, default=50,
                        help='The vertical and horizontal stride when we are sampling patches from the volume.' +
                             'The smaller the better, but the slower the training is.')
    parser.add_argument('--patch_size', nargs='?', type=int, default=99,
                        help='The size of each patch')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=False,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Whether to use data augmentation.')
    parser.add_argument('--class_weights', nargs='?', type=bool, default=False,
                        help='Whether to use class weights to reduce the effect of class imbalance')
    parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
    parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
    parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
    parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
    parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
    parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
    parser.add_argument('--test', action='store_true', default=False,
                    help='training mode')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='Optimiser to use [\'sgd, adam, adadelta\']')
    parser.add_argument('--__lambda', type=str, default="constant_1",
                        help='Weightening factor to use for coral [\'epoch_based_{value}, constant_{value}\']')
    parser.add_argument('--backbone', type=str, default="DeepLab_ResNet9",
                        help='Backbone for DeepCORAL [\'DeepLab_ResNet9, SeismicNet\']')
    parser.add_argument('--normalize_loss', action='store_true', default=False,
                        help='Normalize loss from 0 to 1')

    args = parser.parse_args()
    
    val_fname = pjoin('Results', 'DA_' + args.backbone + '_' + str(args.base_lr) + '_' + str(args.netherlands_batch_size) + '_' + str(args.canada_batch_size) + '_' + args.optim + '_' + args.exp)
    
    os.mkdir(val_fname)
    os.mkdir(val_fname + "/Tensorboard_Records")
    os.mkdir(val_fname + "/metrics")
    os.mkdir(val_fname + "/metrics" + "/confusion_matrix")
    
    train(args)
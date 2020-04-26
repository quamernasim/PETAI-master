import argparse
from datetime import datetime
import itertools
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from tqdm import tqdm
from core.models import SeismicNet4EncoderASPP
import core.loss
import torchvision.utils as vutils
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
from core.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb,AverageMeter, inter_and_union
from PIL import Image
from torchvision import transforms 
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
from core import utils

# Fix the random seeds: 
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def split_train_val(args, per_val=0.1):
    # create inline and crossline pacthes for training and validation:
    loader_type = 'patch'
    labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
    iline, xline, depth = labels.shape

    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline-args.stride, args.stride)
    vert_locations = range(0, depth-args.stride, args.stride)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = ['i_'+str(i)+'_'+str(j)+'_'+str(k)
                        for j, k in locations]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline-args.stride, args.stride)
    vert_locations = range(0, depth-args.stride, args.stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = ['x_'+str(i)+'_'+str(j)+'_'+str(k)
                        for i, k in locations]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    # write to files to disK:
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
def to_3_channels(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #filler = tf.zeros([32,1,99,99], tf.int32)
    size_tensor = x.size()
    filler = torch.zeros((size_tensor[0],size_tensor[1],size_tensor[2],size_tensor[3]), dtype=torch.float).to(device)
    return torch.cat((x, filler, filler), dim=1)

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate the train and validation sets for the model:
    split_train_val(args, per_val=args.per_val)

    current_time = datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join('runs', current_time +
                           "_{}".format(args.arch))
    writer = SummaryWriter(log_dir=log_dir)
    # Setup Augmentations
    if args.aug:
        data_aug = Compose(
            [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
    else:
        data_aug = None

    train_set = PatchLoader(is_transform=True,
                            split='train',
                            stride=args.stride,
                            patch_size=args.patch_size,
                            augmentations=data_aug)

    # Without Augmentation:
    val_set = PatchLoader(is_transform=True,
                          split='val',
                          stride=args.stride,
                          patch_size=args.patch_size)

    n_classes = train_set.n_classes

    trainloader = data.DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  num_workers=1,
                                  shuffle=True)
    valloader = data.DataLoader(val_set,
                                batch_size=args.batch_size,
                                num_workers=1)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)
   

    # Setup Model edited by Tannistha
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = getattr(SeismicNet4EncoderASPP, 'seismicnet')()
            checkpoint = torch.load(args.resume)
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
            model.train()
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        model = getattr(SeismicNet4EncoderASPP, 'seismicnet')()

    # Use as many GPUs as we can
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    model = model.to(device)  # Send to GPU
    #state_dict = {k[7:]: v for k, v in model['state_dict'].items() if 'tracked' not in k}
    #model.load_state_dict(model['state_dict'])

    # PYTROCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU,

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.Adadelta(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
     ### edited by Tannistha to work with new optimizer
    if args.train:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    
        #optimizer = torch.optim.SGD(model.parameters(),lr=args.base_lr, weight_decay=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=0.0001, amsgrad=True)
        start_epoch = 0

    loss_fn = core.loss.cross_entropy

    if args.class_weights:
        # weights are inversely proportional to the frequency of the classes in the training set
        class_weights = torch.tensor(
            [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
    else:
        class_weights = None

    #best_iou = args.best_iou
    best_mca = args.best_mca
    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
    
    for arg in vars(args):
        text = arg + ': ' + str(getattr(args, arg))
        writer.add_text('Parameters/', text)
        
    model_fname = 'data/SeismicNet_' + str(args.base_lr) + '_batch_size_' + str(args.batch_size) + '_' + args.exp + '_epoch_%d.pth'
    val_fname = 'val_SeismicNet_lr_' + str(args.base_lr) + '_batch_size_' + str(args.batch_size) + '_' + args.exp
    
    for epoch in range(start_epoch, args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, total_iteration = 0, 0

        for i, (images, labels) in enumerate(trainloader):
            
            image_original, labels_original = images, labels
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics.update(gt, pred)

            loss = loss_fn(input=outputs, target=labels, weight=class_weights)

            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()

            if args.clip != 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            
            total_iteration = total_iteration + 1

            if (i) % 20 == 0:
                print('epoch: {0}/{1}\t\t'
                  'iter: {2}/{3}\t\t'
                  'training Loss:{4:.4f}'.format(epoch + 1, args.n_epoch, i + 1, len(trainloader), loss.item()))

            numbers = [0]
            if i in numbers:
                # number 0 image in the batch
                tb_original_image = vutils.make_grid(
                    image_original[0][0], normalize=True, scale_each=True)
                writer.add_image('train/original_image',
                                 tb_original_image, epoch + 1)

                labels_original = labels_original.numpy()[0]
                correct_label_decoded = train_set.decode_segmap(np.squeeze(labels_original))
                writer.add_image('train/original_label',np_to_tb(correct_label_decoded), epoch + 1)
                out = F.softmax(outputs, dim=1)

                # this returns the max. channel number:
                prediction = out.max(1)[1].cpu().numpy()[0]
                # this returns the confidence:
                confidence = out.max(1)[0].cpu().detach()[0]
                tb_confidence = vutils.make_grid(
                    confidence, normalize=True, scale_each=True)

                decoded = train_set.decode_segmap(np.squeeze(prediction))
                writer.add_image('train/predicted', np_to_tb(decoded), epoch + 1)
                writer.add_image('train/confidence', tb_confidence, epoch + 1)

                unary = outputs.cpu().detach()
                unary_max = torch.max(unary)
                unary_min = torch.min(unary)
                unary = unary.add((-1*unary_min))
                unary = unary/(unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel]
                    tb_channel = vutils.make_grid(
                        decoded_channel, normalize=True, scale_each=True)
                    writer.add_image(f'train_classes/_{class_names[channel]}', tb_channel, epoch + 1)

        # Average metrics, and save in writer()
        loss_train /= total_iteration
        score, class_iou = running_metrics.get_scores()
        writer.add_scalar('train/Pixel Acc', score['Pixel Acc: '], epoch+1)
        writer.add_scalar('train/Mean Class Acc',
                          score['Mean Class Acc: '], epoch+1)
        writer.add_scalar('train/Freq Weighted IoU',
                          score['Freq Weighted IoU: '], epoch+1)
        writer.add_scalar('train/Mean_IoU', score['Mean IoU: '], epoch+1)
        running_metrics.reset()
        writer.add_scalar('train/loss', loss_train, epoch+1)
        
        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval()
                loss_val, total_iteration_val = 0, 0

                for i_val, (images_val, labels_val) in enumerate(valloader):
                    image_original, labels_original = images_val, labels_val
                    images_val, labels_val = images_val.to(
                        device), labels_val.to(device)
                    #image_val = to_3_channels(images_val)
                    outputs_val = model(images_val)
                    #outputs_val = model(image_val)
                    pred = outputs_val.detach().max(1)[1].cpu().numpy()
                    gt = labels_val.detach().cpu().numpy()

                    running_metrics_val.update(gt, pred)

                    loss = loss_fn(input=outputs_val, target=labels_val)
                    
                    loss_val += loss.item()

                    total_iteration_val = total_iteration_val + 1

                    if (i_val) % 20 == 0:
                        print("Epoch [%d/%d] validation Loss: %.4f" %
                              (epoch+1, args.n_epoch, loss.item()))

                    numbers = [0]
                    if i_val in numbers:
                        # number 0 image in the batch
                        tb_original_image = vutils.make_grid(
                            image_original[0][0], normalize=True, scale_each=True)
                        writer.add_image('val/original_image',
                                         tb_original_image, epoch)
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = train_set.decode_segmap(
                            np.squeeze(labels_original))
                        writer.add_image('val/original_label',
                                         np_to_tb(correct_label_decoded), epoch + 1)

                        out = F.softmax(outputs_val, dim=1)

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0]
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = vutils.make_grid(
                            confidence, normalize=True, scale_each=True)

                        decoded = train_set.decode_segmap(
                            np.squeeze(prediction))
                        writer.add_image('val/predicted', np_to_tb(decoded), epoch + 1)
                        writer.add_image('val/confidence',
                                         tb_confidence, epoch + 1)

                        unary = outputs.cpu().detach()
                        unary_max, unary_min = torch.max(
                            unary), torch.min(unary)
                        unary = unary.add((-1*unary_min))
                        unary = unary/(unary_max - unary_min)

                        for channel in range(0, len(class_names)):
                            tb_channel = vutils.make_grid(
                                unary[0][channel], normalize=True, scale_each=True)
                            writer.add_image(
                                f'val_classes/_{class_names[channel]}', tb_channel, epoch + 1)
                loss_val /= total_iteration_val
                score, class_iou = running_metrics_val.get_scores()
                
                pd.DataFrame([running_metrics_val.get_scores()[0]["Pixel Acc: "]]).to_csv(os.path.join(val_fname, "metrics", "pixel_acc.csv"), index=False, mode='a', header=(i==0))
                pd.DataFrame([running_metrics_val.get_scores()[0]["Mean Class Acc: "]]).to_csv(os.path.join(val_fname, "metrics", "mean_class_acc.csv"),index=False, mode='a', header=(i==0))
                pd.DataFrame([running_metrics_val.get_scores()[0]["Freq Weighted IoU: "]]).to_csv(os.path.join(val_fname, "metrics", "freq_weighted_iou.csv"),index=False, mode='a', header=(i==0))
                pd.DataFrame([running_metrics_val.get_scores()[0]["Mean IoU: "]]).to_csv(os.path.join(val_fname, "metrics", "mean_iou.csv"), index=False, mode='a', header=(i==0))
                
                cname = os.path.join(val_fname, "metrics", "confusion_matrix", "confusion_matrix_" + str(epoch + 1) + ".csv")
                pd.DataFrame(running_metrics_val.get_scores()[0]["confusion_matrix"]).to_csv(cname, index=False)
                
                pd.DataFrame(running_metrics_val.get_scores()[0]["Class Accuracy: "].reshape((1, 6)), columns=[0, 1, 2, 3, 4, 5]).to_csv(os.path.join(val_fname, "metrics", "class_acc.csv"), index=False, mode = "a", header = (i == 0))
                pd.DataFrame(running_metrics_val.get_scores()[1], columns=[0, 1, 2, 3, 4, 5], index=[0]).to_csv(os.path.join(val_fname, "metrics", "cls_iu.csv"), mode = "a", header = (i == 0))
                

                writer.add_scalar(
                    'val/Pixel Acc', score['Pixel Acc: '], epoch+1)
                writer.add_scalar('val/Mean IoU', score['Mean IoU: '], epoch+1)
                writer.add_scalar('val/Mean Class Acc',
                                  score['Mean Class Acc: '], epoch+1)
                writer.add_scalar('val/Freq Weighted IoU',
                                  score['Freq Weighted IoU: '], epoch+1)

                writer.add_scalar('val/loss', loss_val, epoch+1)
                running_metrics_val.reset()

                #if score['Mean IoU: '] >= best_iou:
                if score['Mean Class Acc: '] >= best_mca:
                    #best_iou = score['Mean IoU: ']
                    best_mca = score['Mean Class Acc: ']
                    model_dir = os.path.join(
                        log_dir, f"{args.arch}_model.pkl")
                    #torch.save(model, model_dir)

                    torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),}, model_fname % (epoch + 1))


        else:  # validation is turned off:
            # just save the latest model:
            if (epoch+1) % 5 == 0:
                model_dir = os.path.join(
                    log_dir, f"{args.arch}_ep{epoch+1}_model.pkl")
                #torch.save(model, model_dir)
                torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),}, model_fname % (epoch + 1))
        
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='SeismicNet',
                        help='Architecture to use [\'SeismicNet, DeConvNet, DeConvNetSkip\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=101,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                        help='Batch Size')
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
    parser.add_argument('--best_mca', nargs='?', type=float, default=-100.0,
                        help='best mca if resume True')
                    

    args = parser.parse_args()
    
    val_fname = 'val_SeismicNet_lr_' + str(args.base_lr) + '_batch_size_' + str(args.batch_size) + '_' + args.exp
    
    os.mkdir(val_fname)
    os.mkdir(val_fname + "/metrics")
    os.mkdir(val_fname + "/metrics" + "/confusion_matrix")
    
    train(args)

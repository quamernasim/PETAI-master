import argparse
from datetime import datetime
import itertools
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from tqdm import tqdm
from core.models import deeplab
import core.loss
import torchvision.utils as vutils
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
from core.data_loader import *
from core.metrics import runningScore
from core.models import get_model
from core.utils import np_to_tb,AverageMeter, inter_and_union
from PIL import Image
from torchvision import transforms 
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


# Fix the random seeds: 
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def to_3_channels(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #filler = tf.zeros([32,1,99,99], tf.int32)
    size_tensor = x.size()
    filler = torch.zeros((size_tensor[0],size_tensor[1],size_tensor[2],size_tensor[3]), dtype=torch.float)
    return torch.cat((x, filler, filler), dim=1)
def patch_label_2d(model, img, patch_size, stride):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size/2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0)

    num_classes = 6      
    output_p = torch.zeros([1, num_classes, h+2*ps, w+2*ps])

    # generate output:
    for hdx in range(0, h-patch_size+ps, stride):
        for wdx in range(0, w-patch_size+ps, stride):
            patch = img_p[hdx + ps: hdx + ps + patch_size,
                          wdx + ps: wdx + ps + patch_size]
            patch = patch.unsqueeze(dim=0)  # channel dim
            patch = patch.unsqueeze(dim=0)  # batch dim
            #patch_img = to_3_channels(patch)

            assert (patch.shape == (1, 1, patch_size, patch_size))
            # edited by Tannistha
            #assert (patch_img.shape == (1, 3, patch_size, patch_size))

            model_output = model(patch)
            #model_output = model(patch_img)
            output_p[:, :, hdx + ps: hdx + ps + patch_size, wdx + ps: wdx +
                     ps + patch_size] += torch.squeeze(model_output.detach().cpu())

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


def test(args):
    model = getattr(deeplab, 'resnet18')(
        pretrained=(args.scratch),
        num_classes=6,
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #log_dir, model_name = os.path.split(args.model_path)
    torch.cuda.set_device(args.gpu)
    model = nn.DataParallel(model).cuda()
    #model = model.cuda()
    model.eval()
    checkpoint = torch.load(args.model_path)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    #writer = SummaryWriter(log_dir=log_dir)
    writer = SummaryWriter('run')

    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    #splits = [args.split if 'both' not in args.split else 'test1', 'test2']
    splits = [args.split if 'both' not in args.split else 'test1']
    print(splits)
    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape

        if args.inline:
            i_list = list(range(irange))
            i_list = ['i_'+str(inline) for inline in i_list]
        else:
            i_list = []

        if args.crossline:
            x_list = list(range(xrange))
            x_list = ['x_'+str(crossline) for crossline in x_list]
        else:
            x_list = []

        list_test = i_list + x_list
        

        file_object = open(
            pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()
        test_set = SectionLoader(is_transform=True,
                                 split=split,
                                 augmentations=None)
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      num_workers=4,
                                      shuffle=False)

        running_metrics_split = runningScore(n_classes)
        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for i, (images, labels) in enumerate(test_loader):
                print(f'split: {split}, section: {i}')
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels
                

                outputs = patch_label_2d(model=model,
                                         img=images,
                                         patch_size=args.train_patch_size,
                                         stride=args.test_stride)

                pred = outputs.detach().max(1)[1].numpy()
                gt = labels.numpy()
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)
                print(type(gt), type(pred))

                numbers = [0, 99, 149, 399, 499]
                if i in numbers:
                    tb_original_image = vutils.make_grid(
                        image_original[0][0], normalize=True, scale_each=True)
                    writer.add_image('original_image',
                                     tb_original_image, i)
                    #torchvision.transforms.ToPILImage()(tb_original_image) 
                    my_string = 'image_original_' + str(i)
                    original_image = tb_original_image.permute(1, 2, 0).numpy()
                    fig, ax = plt.subplots(figsize=(14, 8))
                    ax.imshow(original_image)
                    plt.savefig("test/original_image/{}.jpg".format(my_string)) #, img)
                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(
                        np.squeeze(labels_original))
                    writer.add_image('original_label',
                                     np_to_tb(correct_label_decoded), i)
                    
                    fig, ax1 = plt.subplots(figsize=(14, 8))
                    ax1.imshow(correct_label_decoded)
                    my_string1 = 'correct_label_' + str(i)
                    plt.savefig("test/original_label/{}.jpg".format(my_string1)) #, img)
                    out = F.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = vutils.make_grid(
                        confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(np.squeeze(prediction))
                    writer.add_image('predicted', np_to_tb(decoded), i)
                    fig, ax2 = plt.subplots(figsize=(14, 8))
                    ax2.imshow(decoded)
                    my_string2 = 'predicted_' + str(i)
                    plt.savefig("test/predicted/{}.jpg".format(my_string2)) #, img)
                    writer.add_image('test/confidence', tb_confidence, i)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='SeismicNet',
                        help='Architecture to use [\'SeismicNet, DeConvNet, DeConvNetSkip\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=101,
                        help='# of the epochs')
    parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
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
                    help='resnet18 beta')
    parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
    parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
    parser.add_argument('--test', action='store_true', default=False,
                    help='training mode')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--train_patch_size', nargs='?', type=int, default=99,
                        help='The size of the patches that were used for training.'
                        'This must be correct, or will cause errors.')
    parser.add_argument('--test_stride', nargs='?', type=int, default=10,
                        help='The size of the stride of the sliding window at test time. The smaller, the better the results, but the slower they are computed.')
    parser.add_argument('--model_path', nargs='?', type=str, default='path/to/model.pkl',
                        help='Path to the saved model')                    

    args = parser.parse_args()
    test(args)

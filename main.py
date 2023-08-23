import sys
import time
import clip
import torch
import torch.nn as nn

import utils
from clip.clip import _transform
import argparse
from networks import LinearClassifier, DiscoveryMechanism
import tensorboard_logger as tlogger
from utils import *
from data_utils import get_loaders
from torch.utils.tensorboard import SummaryWriter
import itertools

parser = argparse.ArgumentParser(description='Settings for the Concept Discovery Model')
#
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--clip_version", type=str, default="ViT-B/16")
parser.add_argument("--concept_name", type=str, default="cifar100")
parser.add_argument("--device", type=str, default="cuda",
                    help="Which device to use")
parser.add_argument("--save_dir", type=str, default='saved_models',
                    help="where to save the results")
parser.add_argument('--load_similarities', action='store_true',
                    help='load the precomputed similarities for the given dataset')
parser.add_argument('--compute_similarities', action='store_true',
                    help='compute, save and use similarities for the given dataset')
parser.add_argument('--optimizer', default='adam', help='the considered optimizer')
parser.add_argument('--print_freq', type=int, default=100,
                    help='print frequency')
# training args
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs')
parser.add_argument('--num_workers', type=int, default=8,
                    help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay value')
parser.add_argument("--batch_size", type=int, default=2000,
                    help="The training batch size")
parser.add_argument('--save_freq', type=int, default=50,
                    help='save frequency')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint to resume training')

parser.add_argument('--discovery', action='store_true', help='Use data driven concept discovery')
parser.add_argument('--eval', action='store_true', help='Load and eval the models only')
parser.add_argument('--topk', action='store_true', help='Use topk concepts')
parser.add_argument('--ntopk', default='0.1', help='topk percentage of concepts (if topk active)')


def set_clip_model(args, device='cuda'):
    # Load the model
    clip_model, _ = clip.load(args.clip_version, device)
    patch_transform = _transform(clip_model.visual.input_resolution, pil_flag=False, norm_flag=True)

    return clip_model, patch_transform


def train(args, loader, clip_model, concept_set, classifier, criterion, optimizer, epoch, preprocess, disc):
    btime, dtime, losses, top1 = [AverageMeter() for _ in range(4)]
    start_time = time.time()

    for batch, (images, labels) in enumerate(loader):
        dtime.update(time.time() - start_time)
        labels = labels.type(torch.LongTensor)  # casting to long
        images = images.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            if not args.load_similarities:
                text_features = get_clip_text_features(clip_model, concept_set)
                feats = clip_model.encode_image(preprocess(images).to(device))
            else:
                text_features = concept_set
                feats = images

            # normalize and compute similarity (batch size x num_concepts)
            text_features = normalize(text_features, dim=-1)
            feats = normalize(feats, dim=-1)
            S = (feats @ text_features.T) if not args.only_images else images

        # this is the main logic, classifier is num_concepts x classes
        # you can select three modes: concept discovery, classification with topK concepts and standard
        kl = 0.
        if args.discovery:
            z_samples, kl = disc(images)
            output = classifier(S, mask=z_samples)

        elif args.topk:
            val, ind = torch.topk(S, int(args.ntopk * S.shape[1]), dim=1, largest=True)
            masked_similarity = torch.zeros_like(S)
            masked_similarity.scatter_(1, ind, val)
            output = classifier(masked_similarity)
        else:
            output = classifier(S)

        # minimize the cross entropy loss
        loss = criterion(output, labels) + args.kl_scale * kl

        losses.update(loss.item(), output.shape[0])
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], output.shape[0])

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        btime.update(time.time() - start_time)
        start_time = time.time()

        # print info
        if (batch + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch + 1, len(train_loader), batch_time=btime,
                data_time=dtime, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(args, loader, clip_model, concept_set, classifier, criterion, preprocess, disc, num_samples):
    btime, losses, top1 = [AverageMeter() for _ in range(3)]
    avg_perc_active = 0. 
    with torch.no_grad():
        end = time.time()
        for batch, (images, labels) in enumerate(loader):
            labels = labels.type(torch.LongTensor)  # casting to long

            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            with torch.no_grad():
                if not args.load_similarities:
                    text_features = get_clip_text_features(clip_model, concept_set)
                    feats = clip_model.encode_image(preprocess(images).to(device))
                else:
                    text_features = concept_set
                    feats = images

                # normalize and compute similarity (batch size x num_concepts)
                text_features = normalize(text_features, dim=-1)
                feats = normalize(feats, dim=-1)
                S = (feats @ text_features.T) if not args.only_images else images

            if args.discovery:
                output = 0.
                for _ in range(num_samples):
                    z, _ = disc(images, probs_only=True)
                    output += classifier(S, z) / num_samples
                    avg_perc_active += (z > 0.01).cpu().numpy().sum(1).mean()

            elif args.topk:
                val, ind = torch.topk(S, int(args.ntopk * S.shape[1]),
                                      dim=1, largest=True)
                masked_similarity = torch.zeros_like(S)
                masked_similarity.scatter_(1, ind, val)
                output = classifier(masked_similarity)

            else:
                output = classifier(S)

            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), output.shape[0])
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], output.shape[0])

            # measure elapsed time
            btime.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Avg Perc Act {avg:.4f}'.format(
                    batch, len(loader), batch_time=btime,
                    loss=losses, top1=top1, avg=avg_perc_active/batch))

        print(' Validation Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg, avg_perc_active/batch


if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
    args.device = device

    concepts = [args.dataset]
    kl_scales = [1.]
    lrs = [0.001]
    priors = [0.005]
    mults = [1.0]
    discover = [True]
    cversions = ['ViT-B/16']

    # if true perform classification using topk concepts per image
    topk = [False]
    topk_perc = [0.05, 0.1, 0.2, 0.4]

    # if true classify using only the clip image embeddings
    only_images = False

    params = [concepts, kl_scales, lrs, discover, priors, mults, cversions, topk, topk_perc]
    for concept, kl_scale, lr, discover_flag, prior, mult, cversion, topk, ntopk in itertools.product(*params):

        args.concept_name = concept
        args.discovery = discover_flag
        args.learning_rate = lr
        args.prior = prior
        args.kl_scale = kl_scale
        args.multiplier_lr = mult
        args.topk = topk
        args.ntopk = ntopk
        args.only_images = only_images
        args.clip_version = cversion

        print(args)

        if args.compute_similarities or not args.load_similarities:
            clip_model, preprocess = set_clip_model(args)
        else:
            clip_model, preprocess = None, None

        train_loader, val_loader, classes, concept_set = get_loaders(args)
        num_classes, num_concepts = len(classes), len(concept_set)

        if args.compute_similarities:
            with torch.no_grad():
                texts = torch.cat([clip.tokenize(f"{c}") for c in concept_set]).to(device)
                compute_and_save_features(args, clip_model, train_loader, texts,
                                          preprocess, device=device)
                compute_and_save_features(args, clip_model, val_loader, texts,
                                          preprocess, device=device, val=True)
                sys.exit()

        if only_images and (topk or discover_flag):
            raise ValueError('Can\'t use only the image embeddings with discovery....')


        if not discover_flag:

            if not topk and ntopk != 0.05:
                continue
            if kl_scale != 1e-4 or prior != 0.01 or mult != 5.:
                continue
        else:
            if topk or ntopk != 0.05:
                continue


        results_dir = utils.get_feature_dir(args)

        spec_dir = results_dir[:-17] + 'only_images/' * only_images + '{}_concept_set/mask_{}/{}_{}_multiplier_{}/'.format(
            args.concept_name,
            args.discovery,
            args.optimizer,
            str(args.learning_rate).replace(
                '.', '_'),
            str(args.multiplier_lr).replace(
                '.', '_')
        )
        if args.discovery:
            spec_dir += 'kl_scale_{}_prior_{}/'.format(str(args.kl_scale).replace('.', ''),
                                                       str(args.prior).replace('.', ''))
        if args.topk:
            spec_dir += 'top_k_perc_{}/'.format(str(args.ntopk).replace('.', ''))

        if os.path.exists(spec_dir) and not args.eval:
            count = 0
            new_dir = spec_dir
            while os.path.exists(new_dir):
                count += 1
                new_dir = spec_dir[:-1] + '_run_{}/'.format(count)
            spec_dir = new_dir

        os.makedirs(spec_dir, exist_ok=True)

        if not args.eval:
            with open(spec_dir + 'args.txt', 'w') as f:
                f.write(str(args))
            writer = SummaryWriter(spec_dir)

        if not args.load_similarities:
            with torch.no_grad():
                texts = torch.cat([clip.tokenize(f"{c}") for c in concept_set]).to(device)
        else:
            text_dir = get_feature_dir(args)
            text_path = text_dir + '{}_text_features.pt'.format(args.concept_name)
            # the specific concept set does not exist create it
            if not os.path.exists(text_path):
                warnings.warn('!!!! Warning !!!!\n The text features do not exist.\n'
                              'Will create a new file\n'
                              '!!!!!!!!!!!!!!!!!')
                clip_model, _ = set_clip_model(args)
                texts = torch.cat([clip.tokenize(f"{c}") for c in concept_set]).to(device)
                compute_and_save_features(args, clip_model.to(device), None, texts, None, device=device, only_text=True)
            texts = torch.load(text_path).float()

        # get the dimensionality of CLIP through the features in the dataset
        criterion = nn.CrossEntropyLoss().to(device)

        feat_emb = 512 if 'ViT' in args.clip_version else 1024
        classifier = LinearClassifier(num_concepts if not args.only_images else feat_emb, num_classes).to(device)
        disc = DiscoveryMechanism(feat_emb, num_concepts, prior=prior).to(device) if args.discovery else None

        parameters = [classifier.parameters(), disc.parameters()] \
            if args.discovery else classifier.parameters()
        optimizer = get_optimizer(args, parameters)

        # load the checkpoints and eval
        if args.eval:
            if not os.path.exists(spec_dir + 'checkpoint_best.pth.tar'):
                continue
            ckpt = torch.load(spec_dir + 'checkpoint_best.pth.tar')
            classifier.load_state_dict(ckpt['state_dict'])
            disc.load_state_dict(ckpt['disc_state_dict'])

            concept_activation_per_example(val_loader, disc, classifier,
                                           num_concepts, concept_set, classes,
                                           texts, args.dataset, spec_dir)
            continue

        logger = tlogger.Logger(spec_dir + 'logs', flush_secs=10)

        start_time = time.time()
        num_samples = 1 if not args.discovery else 1

        best_acc = 0.
        # print(train_loader[0])
        for epoch in range(args.epochs):

            # training
            classifier.train()
            if args.discovery:
                disc.train()
            loss, acc = train(args, train_loader, clip_model, texts,
                              classifier, criterion, optimizer,
                              epoch, preprocess, disc)

            # validation
            classifier.eval()
            if args.discovery:
                disc.eval()
            val_loss, val_acc, act_perc = validate(args, val_loader, clip_model, texts,
                                                   classifier, criterion,
                                                   preprocess, disc, num_samples)

            log_stats = 'Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, ' \
                        'Val Loss: {:.4f}, Val Acc: {:.4f}, Avg Perc Act: {:.4f}\n'. \
                format(epoch, loss, acc, val_loss, val_acc, act_perc)

            with open(spec_dir + "log.txt", 'a') as f:
                f.write(log_stats)

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'load_similarities': args.load_similarities,
                    'state_dict': classifier.state_dict(),
                    'disc_state_dict': disc.state_dict() if args.discovery else None,
                    'best_acc1': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, filename=spec_dir + 'checkpoint.pth.tar')
                with open(spec_dir + 'best_acc.txt', 'w') as f:
                    f.write('Best acc: {}'.format(best_acc))

            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'load_similarities': args.load_similarities,
                    'state_dict': classifier.state_dict(),
                    'disc_state_dict': disc.state_dict() if args.discovery else None,
                    'best_acc1': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=spec_dir + 'checkpoint.pth.tar')

            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('Accuracy/test', val_acc, epoch)
            writer.add_scalar('Sparsity', act_perc, epoch)
            print('Epoch: {}, Top Validation Acc@1 {:.3f}'.format(epoch, best_acc))

        del optimizer, classifier, disc, logger, criterion, parameters, val_loader, train_loader

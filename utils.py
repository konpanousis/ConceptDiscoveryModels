import torch.optim as optim
import os
import torch
from tqdm import tqdm
import math
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 22})
rc('text', usetex=True)


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


def get_optimizer(args, parameters):
    """
    Set the optimizer for the model; currently supports only Adam.
    :param args: the current setup arguments to get the learning rate and decay.
    :param parameters: the parameters to optimize; either a list of sets of parameters or a single set.

    :return: the optimizer object for training.
    """
    if isinstance(parameters, list):
        new_params = [{
            'params': parameters[0]
        },
            {
                'params': parameters[1],
                'lr': args.learning_rate * args.multiplier_lr,
                'weight_decay': 0.
            }
        ]
    else:
        new_params = parameters
    optimizer = optim.AdamW(new_params, lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    return optimizer


def get_save_names(clip_name, d_probe, concept_set, save_dir):
    clip_save_name = "{}/batches/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))

    return clip_save_name, text_save_name


def get_feature_dir(args, val=False):
    save_dir = args.save_dir + '/{}/{}/embeddings/{}/'.format(args.dataset, args.clip_version.replace('/', '_'),
                                                   'train' if not val else 'val')

    return save_dir

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path_best = filename.split('.')[0] + '_best.pth.tar'
        shutil.copyfile(filename, save_path_best)

def compute_and_save_features(args, model, dataset, text, preprocess,
                              save_frequency=1e6, device="cuda", only_text=False, val=False):
    """

    :param val:
    :param only_text:
    :param args:
    :param model:
    :param dataset:
    :param text:
    :param base_dir:
    :param preprocess:
    :param save_frequency:
    :param device:
    :return:
    """

    save_dir = get_feature_dir(args, val=val)
    os.makedirs(save_dir, exist_ok=True)
    save_name_features = save_dir + 'image_{}_'.format('feats')
    save_name_texts = save_dir + '{}_text_features.pt'.format(args.concept_name)

    if os.path.exists(save_name_features):
        warnings.warn('File for features already exists.')
        return

    text_features = get_clip_text_features(model, text).float()

    if not eval:
        torch.save(text_features, save_name_texts)

    if only_text:
        torch.save(text_features, save_name_texts)
        return

    with torch.no_grad():
        batch_num, save_counter = 1, 0
        feats_enc = []
        labels_batch = []
        for images, labels in tqdm(dataset):

            feats = images
            feats = preprocess(feats)

            features = model.encode_image(feats.to(device)).float()
            feats_enc.append(features.cpu())

            labels_batch.append(labels.cpu())

            if batch_num % save_frequency == 0:
                torch.save((torch.cat(feats_enc, 0), torch.cat(labels_batch, 0)),
                           save_name_features + '_{}.pt'.format(save_counter))
                del feats_enc
                del labels_batch
                feats_enc = []
                labels_batch = []
                save_counter += 1
            batch_num += 1

    # we have some data left on the list
    if feats_enc:
        torch.save((torch.cat(feats_enc, 0).cpu(), torch.cat(labels_batch, 0).cpu()),
                   save_name_features + '_{}.pt'.format(save_counter))
    # free memory
    del feats_enc
    del labels_batch
    torch.cuda.empty_cache()

    return


def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text) / batch_size)):
            text_features.append(model.encode_text(text[batch_size * i:batch_size * (i + 1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features


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

def normalize(data, dim=-1):
    data /= data.norm(dim=dim, keepdim=True)
    return data

def bin_concrete_sample(a, temperature, eps=1e-8):
    """"
    Sample from the binary concrete distribution
    """

    U = torch.rand_like(a).clamp(eps, 1. - eps)
    L = torch.log(U) - torch.log(1. - U)
    X = torch.sigmoid((L + a) / temperature)

    return X


def concept_activation_per_example(loader, disc, classifier, num_concepts,
                                   concepts, classes, text_features, dataset,
                                   base_dir='saved_models', ):
    disc.eval()
    classifier.eval()

    data = loader.dataset
    inds = np.random.choice(np.arange(10, 500, 1), 3, replace=False)
    results_dir = base_dir + 'post_analysis/'
    os.makedirs(results_dir, exist_ok=True)

    indices = 'imagenet_indices.csv' if 'imagenet' in dataset else 'cub_indices.csv'
    with open(indices, 'r') as f:
        csvreader = csv.reader(f)
        count = 0
        fpaths = {}
        for row in csvreader:
            if count == 0:
                count = 1
                continue
            fpaths[int(row[0])] = row[1]

    # for each example get the decision, get the weights corresponding to the class
    # and then get the concept contribution from that
    with torch.no_grad():
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for ind in inds:

            cur_feats, cur_label = data[ind]

            cur_feats = cur_feats.to('cuda')
            cur_label = cur_label.to('cuda')
            mask, _ = disc(cur_feats, probs_only=True)
            cur_feats /= cur_feats.norm(dim=-1, keepdim=True)
            similarity = (cur_feats @ text_features.T)

            pred = torch.nn.functional.softmax(classifier(similarity, mask=mask), dim=-1)

            class_ind = torch.argmax(pred)
            class_name = classes[class_ind]
            class_name = ''.join([i for i in class_name if (not i.isdigit()) and (i != '.')])
            class_name = class_name.replace('_', ' ')
            true_class = classes[int(cur_label.cpu().numpy())]
            true_class = ''.join([i for i in true_class if not i.isdigit() and (i != '.')])
            true_class = true_class.replace('_', ' ')

            if class_name != true_class:
                continue

            spec_dir = results_dir + 'ind_{}/'.format(ind)
            os.makedirs(spec_dir, exist_ok=True)

            # get the original image if cub or imagenet
            if dataset in ['imagenet', 'cub']:
                fpath = fpaths[ind]
                fname = fpath.split('/')[-1]

                shutil.copyfile(fpath, spec_dir + fname)

            mask[torch.where(mask < 0.01)] = 0.

            # get the relevant weights and use the mask to filter
            # the mask should be probs to reflect the contribution
            rel_weights = (classifier.W[:, class_ind] * mask).cpu().numpy()
            rel_bias = classifier.bias[class_ind].cpu().numpy()
            rel_conf = pred[class_ind].cpu().numpy()
            mask = mask.cpu().numpy().astype(np.float32)
            similarity = similarity.cpu().numpy()
            contribution = similarity * rel_weights
            sparsity = (mask > 0.01).sum() / num_concepts

            sort_inds = np.argsort(contribution)[::-1]

            with open(spec_dir + 'stats.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])
                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                     rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', (mask > 0.).sum() / num_concepts])

            with open(spec_dir + 'stats_only_active.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])

                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    if mask[sort_inds[i]] > 0.01:
                        writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                         rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', sparsity])

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig, ax = plt.subplots(layout='constrained', figsize=(18, 6))

            # Example data
            concepts_top = (concepts[sort_inds[0]],
                            concepts[sort_inds[1]],
                            concepts[sort_inds[2]],
                            concepts[sort_inds[3]],
                            'NOT ' * (contribution[sort_inds[-4]] < 0) + concepts[sort_inds[-4]],
                            'NOT ' * (contribution[sort_inds[-3]] < 0) + concepts[sort_inds[-3]],
                            'NOT ' * (contribution[sort_inds[-2]] < 0) + concepts[sort_inds[-2]],
                            'NOT ' * (contribution[sort_inds[-1]] < 0) + concepts[sort_inds[-1]]
                            )
            y_pos = np.arange(len(concepts_top))
            contribution = np.abs(contribution)
            contrib = contribution[sort_inds[:4]].tolist() + contribution[sort_inds[-4:]].tolist()
            colors = ['darkkhaki'] * 4 + ['darksalmon'] * 4
            hbars = ax.barh(y_pos, contrib, align='center',
                            height=0.5, color=colors)
            ax.set_yticks(y_pos, labels=concepts_top)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Concept Contribution')
            ax.set_title('Pred: {} - Conf: {:.5f} - Sparsity: {:.2f}\%'.format(class_name, rel_conf, 100 * sparsity))
            ax.bar_label(hbars, fmt='%.4f')
            ax.margins(x=0.1)

            plt.tight_layout()
            plt.savefig(spec_dir + 'contribution.pdf')  # , bbox_inches="tight")
            plt.close(fig)

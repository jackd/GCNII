from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--simplified', action='store_true', default=False, help='use simplified model.')
parser.add_argument('--cg', action='store_true', default=False, help='Use CG PageRank.')
parser.add_argument('--hist', nargs='*', type=int, help='conv layer indices to plot histograms of', default=[])
parser.add_argument('--repeats', type=int, default=10, help='number of runs')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

hist = args.hist
if hist and (args.simplified or args.cg):
    raise RuntimeError("`hist` and `simplified` not simultaneously supported")
# Load data
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(args.seed,cudaid,checkpt_file)

model = GCNII(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                simplified=args.simplified,
                cg=args.cg,
                variant=args.variant).to(device)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

accs = []
for _ in range(args.repeats):
    model.reset_parameters()
    optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)

    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train()
        loss_val,acc_val = validate()
        if(epoch+1)%100 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
            if hist:
                def get_values(h:int):
                    layer = model.convs[h]
                    scaled_weights = layer.weight.detach().cpu().numpy() * layer.theta
                    outputs = layer.output.detach().cpu().numpy()
                    return scaled_weights, outputs
                weights, outputs = zip(*(get_values(h) for h in hist))
                num_negatives = [(torch.count_nonzero(layer.output < 0) / layer.output.size().numel()).detach().cpu().numpy() for layer in model.convs]
                # tail_weight = [-layer.output[layer.output < 0].sum().detach().cpu().numpy() for layer in model.convs]
                negative_mean = [-layer.output[layer.output < 0].mean().detach().cpu().numpy() for layer in model.convs]
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        acc = test()[1]
    accs.append(acc)

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))


    if hist:
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(1, 2)

        weight_min = min(w.min() for w in weights)
        weight_max = max(w.max() for w in weights)
        output_min = min(o.min() for o in outputs)
        output_max = max(o.max() for o in outputs)
        # output_max = np.quantile(np.concatenate(outputs, axis=0).reshape(-1), 0.99)
        # output_max = 0.002
        for h, weight, output in zip(hist, weights, outputs):
            if h < 0:
                h += len(model.convs)
            label = f"$k = {h+1}$"
            ax0.hist(weight.reshape(-1), label=label, alpha=0.4, range=(weight_min, weight_max), bins=200, cumulative=False, density=False)
            # ax1.hist(output.reshape(-1), label=label, alpha=0.4, range=(output_min, output_max), bins=10000, cumulative=True, density=True)
            output = output.reshape(-1)
            output.sort()
            ax1.fill_between(output, np.arange(1, output.shape[0]+1) / (output.shape[0]+1), label=label, alpha=0.4)
        ax0.legend()
        ax1.legend()
        ax0.set_yticks([])
        ax0.set_xlim(-0.0004, 0.0004)
        ax0.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax0.set_title(r"$\beta_k \theta^{(k)}$")
        ax1.set_title("Cumulative Preactivations")
        ax1.set_xlim(-0.001, 0.002)
        ax1.set_ylim(0, 0.5)
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        fig.tight_layout()

        fig = plt.figure()
        ax2 = plt.gca()
        color = 'tab:blue'

        x = np.arange(1, len(num_negatives)+1)
        ax2.plot(x, num_negatives, color=color)
        ax2.set_xlabel("$k$")
        ax2.set_ylabel("Negative proportion", color=color)
        ax2.set_ylim(0, 0.08)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax2.twinx()
        color = 'tab:red'
        ax3.plot(x, negative_mean, color=color)
        ax3.set_ylabel("Mean negative preactivation magnitude", color=color)
        ax3.set_ylim(0, np.max(negative_mean)*1.1)
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax3.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.show()


print(f"acc = {np.mean(accs)} ± {np.std(accs)}")

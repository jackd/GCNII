import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_sparse import SparseTensor
from torch_cg import cg_batch


def _get_batch_fn(A: SparseTensor):
    def fn(x):
        return torch.spmm(A, x.squeeze(0)).unsqueeze(0)
    return fn


class SparseCG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_st, B):
        ctx.save_for_backward(A_st)
        X, _ = cg_batch(_get_batch_fn(A_st), B.unsqueeze(0), maxiter=100)
        return X.squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        A_st, = ctx.saved_tensors
        grad_B, _ = cg_batch(_get_batch_fn(A_st), grad_output.unsqueeze(0), maxiter=100)
        grad_B = grad_B.squeeze(0)
        grad_A = None
        return grad_A, grad_B

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, simplified=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.simplified = simplified
        if not self.simplified:
            self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()
        self.output = None
        self.theta = None

    def reset_parameters(self):
        if not self.simplified:
            stdv = 1. / math.sqrt(self.out_features)
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):

        hi = torch.spmm(adj, input)
        if self.simplified:
            return (1 - alpha) * hi + alpha * h0
        theta = math.log(lamda/l+1)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        self.output = output
        self.theta = theta
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, simplified=False, cg=False):
        super(GCNII, self).__init__()
        self.cg = cg
        if not self.cg:
            self.convs = nn.ModuleList()
            for _ in range(nlayers):
                self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant, simplified=simplified))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = [] if self.cg else list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        if not self.cg:
            for conv in self.convs:
                conv.reset_parameters()

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        if self.cg:
            # layer_inner = PageRank.apply(adj, layer_inner, 0.1)
            I = SparseTensor.eye(adj.shape[0]).to_torch_sparse_coo_tensor().to(adj.device)
            A_st = I - (1 - self.alpha)*adj
            layer_inner = self.alpha * SparseCG.apply(A_st, layer_inner)
        else:
            for i,con in enumerate(self.convs):
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == '__main__':
    pass







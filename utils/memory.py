import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ContrastMemory(nn.Module):
    """
    Memory buffer that supplies a large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, P, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)  # Use .to(device) instead of .cuda()
        self.P = P
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum, P]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, label_matrix, idx=None):
        """
        Forward pass for ContrastMemory.
        v1: Feature vector for memory bank 1
        v2: Feature vector for memory bank 2
        y: Batch labels (should be a 1D tensor of class indices or one-hot encoded)
        label_matrix: Label similarity matrix
        idx: Optional indices for sampling (default: None)
        """
        batchSize = v1.size(0)

        # If `y` is a one-hot encoded tensor or a tensor with extra dimensions,
        # convert it to class indices (e.g., [batch_size, num_classes] to [batch_size]).
        if y.ndimension() > 1:
            # Cast y to long (or int64) type before applying torch.argmax
            y = torch.argmax(y.to(torch.long), dim=1)

        # Ensure `y` is a 1D tensor of class indices
        y = y.view(-1)  # Reshape to [batch_size]

        # Initialize `idx` if None
        if idx is None:
            idx = torch.zeros((batchSize, self.K + self.P), dtype=torch.long, device=y.device)

        # Assign positive samples to the first column of `idx`
        idx[:, 0] = y

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        P = int(self.params[5].item())

        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + P, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + P, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("Normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("Normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2, self.memory_v1, self.memory_v2


class AliasMethod(object):
    """
    Implements the Alias Method for efficient sampling with many discrete outcomes.
    """
    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)

        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj

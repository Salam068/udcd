# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import numpy as np
# from utils.memory import ContrastMemory
# from utils.losses import uncertainity_loss
#
# eps = 1e-7
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class CRALoss(nn.Module):
#     """
#     Args:
#         opt.s_dim: the dimension of student's feature
#         opt.t_dim: the dimension of teacher's feature
#         opt.feat_dim: the dimension of the projection space
#         opt.nce_k: number of negatives paired with each positive
#         opt.nce_t: the temperature
#         opt.nce_m: the momentum for updating the memory buffer
#         opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
#     """
#
#     def __init__(self, opt):
#         super(CRALoss, self).__init__()
#         self.embed_s = Embed(opt.s_dim, opt.feat_dim)
#         self.embed_t = Embed(opt.t_dim, opt.feat_dim)
#         self.contrast = ContrastMemory(
#             opt.feat_dim, opt.n_data, opt.nce_p, opt.nce_k, opt.nce_t, opt.nce_m)
#
#         self.criterion_t = ContrastLoss(opt.n_data)
#         self.criterion_s = ContrastLoss(opt.n_data)
#
#         self.relation_loss = anchor_relation_loss(opt.nce_t)
#         self.anchor_type = opt.anchor_type
#         self.class_anchor = opt.class_anchor
#
#     def forward(self, f_s, f_t, idx, batch_label, class_index, num_pos, contrast_idx=None):
#         """
#         Args:
#             f_s: the feature of student network, size [batch_size, s_dim]
#             f_t: the feature of teacher network, size [batch_size, t_dim]
#             idx: the indices of these positive samples in the dataset, size [batch_size]
#             contrast_idx: the indices of negative samples, size [batch_size, nce_k]
#         """
#         f_s = self.embed_s(f_s)
#         f_t = self.embed_t(f_t)  # after l2-norm, the norm of f_s and f_t is 1.
#
#         batch_label = torch.argmax(batch_label, axis=1)
#         batch_label_matrix = torch.eq(
#             batch_label.view(-1, 1), batch_label.view(1, -1))
#
#         out_s, out_t, self.memory_s, self.memory_t = self.contrast(
#             f_s, f_t, idx, batch_label_matrix, contrast_idx)
#         s_loss = self.criterion_s(out_s, num_pos)
#         t_loss = self.criterion_t(out_t, num_pos)
#         CCD_loss = s_loss + t_loss
#
#         s_anchors, t_anchors = None, None
#         for i in range(len(class_index)):
#             if self.anchor_type == "center":
#                 img_index = torch.tensor(class_index[i]).cuda()
#
#                 s_anchors_i = torch.index_select(
#                     self.memory_s.cuda(), 0, img_index.view(-1))
#                 s_center_i = torch.mean(
#                     F.relu(s_anchors_i), axis=0, keepdims=True)
#                 s_anchors = s_center_i if s_anchors is None else torch.cat(
#                     (s_anchors, s_center_i), axis=0)
#
#                 t_anchors_i = torch.index_select(
#                     self.memory_t.cuda(), 0, img_index.view(-1))
#                 t_center_i = torch.mean(
#                     F.relu(t_anchors_i), axis=0, keepdims=True)
#                 t_anchors = t_center_i if t_anchors is None else torch.cat(
#                     (t_anchors, t_center_i), axis=0)
#
#             elif self.anchor_type == "class":
#                 img_index = torch.tensor(np.random.permutation(
#                     class_index[i])[0:self.class_anchor]).cuda()
#                 s_anchors_i = torch.index_select(
#                     self.memory_s.cuda(), 0, img_index.view(-1))
#                 s_anchors = s_anchors_i if s_anchors is None else torch.cat(
#                     (s_anchors, s_anchors_i), axis=0)
#
#                 t_anchors_i = torch.index_select(
#                     self.memory_t.cuda(), 0, img_index.view(-1))
#                 t_anchors = t_anchors_i if t_anchors is None else torch.cat(
#                     (t_anchors, t_anchors_i), axis=0)
#
#         relation_loss = self.relation_loss(f_s, s_anchors, f_t, t_anchors)
#         return CCD_loss, relation_loss
#
#
# class anchor_relation_loss(nn.Module):
#     """
#     Compute relation matrix between the batch feature and the anchors (can be centroid features of all classes or
#     randomly sampled features from each class.
#     :param f: batch features. [bs, feat_dim]
#     :param anchors: centroid features of seven classes. [n_anchors, feat_dim]
#     :return: loss between the relation matrix of the student and ema teacher.
#     """
#
#     def __init__(self, T):
#         super(anchor_relation_loss, self).__init__()
#         self.l2norm = Normalize(2)
#         self.T = T
#         self.cos = nn.CosineEmbeddingLoss().cuda()
#         self.target = torch.Tensor([1]).cuda()  # for dissimilarity
#
#     def forward(self, f_s, s_anchors, f_t, t_anchors):
#         s_anchors = self.l2norm(s_anchors)
#         s_relation = torch.div(
#             torch.mm(f_s, s_anchors.clone().detach().T), self.T)  # [bs, n_anchors]
#
#         t_anchors = self.l2norm(t_anchors)
#         t_relation = torch.div(
#             torch.mm(f_t, t_anchors.clone().detach().T), self.T)  # [bs, n_anchors]
#
#         loss = uncertainity_loss(s_relation, t_relation.detach())
#
#         return loss
#
#
# class ContrastLoss(nn.Module):
#     """
#     Supervised contrastive loss
#     """
#
#     def __init__(self, n_data):
#         super(ContrastLoss, self).__init__()
#         self.n_data = n_data
#
#     def forward(self, x, P):
#         bsz = x.shape[0]
#         N = x.size(1) - P
#         # m = N/P
#         m = N
#         # print(m)
#
#         # noise distribution
#         Pn = 1 / float(self.n_data)
#
#         # loss for positive pair
#         P_pos = x.narrow(1, 0, P)
#         log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
#         # print("positive:", log_D1)
#
#         # loss for K negative pair
#         P_neg = x.narrow(1, P, N)
#         log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
#                            P_neg.add(m * Pn + eps)).log_()
#
#         # Using positive samples from the memory bank.
#         # average of the 1 exact pos. sample and (P-1) relax pos. samples.
#         loss = - ((log_D1.squeeze().sum(0) + log_D0.view(-1,
#                   1).repeat(1, P).sum(0)) / bsz).sum(0) / P
#
#         return loss
#
#
# class Embed(nn.Module):
#     def __init__(self, dim_in=1024, dim_out=128):
#         super(Embed, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         self.l2norm = Normalize(2)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         x = self.l2norm(x)
#         return x
#
#
# class Normalize(nn.Module):
#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power
#
#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm)
#         return out
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils.memory import ContrastMemory
from utils.losses import uncertainity_loss
#
# eps = 1e-7
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class CRALoss(nn.Module):
#     def __init__(self, opt):
#         super(CRALoss, self).__init__()
#         self.embed_s = Embed(opt.s_dim, opt.feat_dim).to(device)
#         self.embed_t = Embed(opt.t_dim, opt.feat_dim).to(device)
#         self.contrast = ContrastMemory(
#             opt.feat_dim, opt.n_data, opt.nce_p, opt.nce_k, opt.nce_t, opt.nce_m).to(device)
#
#         self.criterion_t = ContrastLoss(opt.n_data).to(device)
#         self.criterion_s = ContrastLoss(opt.n_data).to(device)
#
#         self.relation_loss = anchor_relation_loss(opt.nce_t).to(device)
#         self.anchor_type = opt.anchor_type
#         self.class_anchor = opt.class_anchor
#
#     def forward(self, f_s, f_t, idx, batch_label, class_index, num_pos, contrast_idx=None):
#         f_s = self.embed_s(f_s)
#         f_t = self.embed_t(f_t)
#
#         batch_label = torch.argmax(batch_label, axis=1)
#         batch_label_matrix = torch.eq(
#             batch_label.view(-1, 1), batch_label.view(1, -1)).to(device)
#
#         out_s, out_t, self.memory_s, self.memory_t = self.contrast(
#             f_s, f_t, idx, batch_label_matrix, contrast_idx)
#         s_loss = self.criterion_s(out_s, num_pos)
#         t_loss = self.criterion_t(out_t, num_pos)
#         CCD_loss = s_loss + t_loss
#
#         s_anchors, t_anchors = None, None
#         for i in range(len(class_index)):
#             img_index = torch.tensor(class_index[i]).to(device)
#
#             if self.anchor_type == "center":
#                 s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
#                 s_center_i = torch.mean(F.relu(s_anchors_i), axis=0, keepdims=True)
#                 s_anchors = s_center_i if s_anchors is None else torch.cat((s_anchors, s_center_i), axis=0)
#
#                 t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
#                 t_center_i = torch.mean(F.relu(t_anchors_i), axis=0, keepdims=True)
#                 t_anchors = t_center_i if t_anchors is None else torch.cat((t_anchors, t_center_i), axis=0)
#
#             elif self.anchor_type == "class":
#                 img_index = torch.tensor(np.random.permutation(
#                     class_index[i])[0:self.class_anchor]).to(device)
#                 s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
#                 s_anchors = s_anchors_i if s_anchors is None else torch.cat((s_anchors, s_anchors_i), axis=0)
#
#                 t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
#                 t_anchors = t_anchors_i if t_anchors is None else torch.cat((t_anchors, t_anchors_i), axis=0)
#
#         relation_loss = self.relation_loss(f_s, s_anchors, f_t, t_anchors)
#         return CCD_loss, relation_loss
#
#
# class anchor_relation_loss(nn.Module):
#     def __init__(self, T):
#         super(anchor_relation_loss, self).__init__()
#         self.l2norm = Normalize(2).to(device)
#         self.T = T
#         self.cos = nn.CosineEmbeddingLoss().to(device)
#         self.target = torch.Tensor([1]).to(device)
#
#     def forward(self, f_s, s_anchors, f_t, t_anchors):
#         s_anchors = self.l2norm(s_anchors)
#         s_relation = torch.div(
#             torch.mm(f_s, s_anchors.clone().detach().T), self.T)
#
#         t_anchors = self.l2norm(t_anchors)
#         t_relation = torch.div(
#             torch.mm(f_t, t_anchors.clone().detach().T), self.T)
#
#         loss = uncertainity_loss(s_relation, t_relation.detach())
#         return loss
#
#
# class ContrastLoss(nn.Module):
#     def __init__(self, n_data):
#         super(ContrastLoss, self).__init__()
#         self.n_data = n_data
#
#     def forward(self, x, P):
#         bsz = x.shape[0]
#         N = x.size(1) - P
#         m = N
#         Pn = 1 / float(self.n_data)
#
#         P_pos = x.narrow(1, 0, P)
#         log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
#
#         P_neg = x.narrow(1, P, N)
#         log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
#                            P_neg.add(m * Pn + eps)).log_()
#
#         loss = - ((log_D1.squeeze().sum(0) + log_D0.view(-1,
#                   1).repeat(1, P).sum(0)) / bsz).sum(0) / P
#
#         return loss
#
#
# class Embed(nn.Module):
#     def __init__(self, dim_in=1024, dim_out=128):
#         super(Embed, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         self.l2norm = Normalize(2).to(device)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         x = self.l2norm(x)
#         return x
#
#
# class Normalize(nn.Module):
#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power
#
#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm)
#         return out

#
# eps = 1e-7
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class CRALoss(nn.Module):
#     def __init__(self, opt):
#         super(CRALoss, self).__init__()
#         self.embed_s = Embed(opt.s_dim, opt.feat_dim).to(device)
#         self.embed_t = Embed(opt.t_dim, opt.feat_dim).to(device)
#         self.contrast = ContrastMemory(
#             opt.feat_dim, opt.n_data, opt.nce_p, opt.nce_k, opt.nce_t, opt.nce_m).to(device)
#
#         self.criterion_t = ContrastLoss(opt.n_data).to(device)
#         self.criterion_s = ContrastLoss(opt.n_data).to(device)
#
#         self.relation_loss = anchor_relation_loss(opt.nce_t).to(device)
#         self.anchor_type = opt.anchor_type
#         self.class_anchor = opt.class_anchor
#
#     def forward(self, f_s, f_t, batch_label, class_index, num_pos, contrast_idx=None):
#         f_s = self.embed_s(f_s)
#         f_t = self.embed_t(f_t)
#
#         # Debugging `batch_label` and `class_index`
#         print(f"Batch label shape: {batch_label.shape}")
#         print(f"Class index type: {type(class_index)}")
#         print(f"Class index content: {class_index}")
#
#         if batch_label.ndimension() > 1:
#             batch_label = torch.argmax(batch_label, dim=1)
#
#         batch_label_matrix = torch.eq(batch_label.view(-1, 1), batch_label.view(1, -1)).to(device)
#
#         if contrast_idx is None:
#             contrast_idx = self.compute_contrast_idx(batch_label_matrix)
#
#         out_s, out_t, self.memory_s, self.memory_t = self.contrast(
#             f_s, f_t, batch_label_matrix, contrast_idx)
#
#         s_loss = self.criterion_s(out_s, num_pos)
#         t_loss = self.criterion_t(out_t, num_pos)
#         CCD_loss = s_loss + t_loss
#
#         # Debugging anchor calculations
#         print(f"Anchor type: {self.anchor_type}")
#         print(f"Class index length: {len(class_index)}")
#
#         s_anchors, t_anchors = None, None
#         for i in range(len(class_index)):
#             # Debugging individual class index
#             print(f"Processing class index {i}")
#             if isinstance(class_index, dict) and i not in class_index:
#                 print(f"Key {i} not found in class_index.")
#                 continue  # Skip missing keys
#
#             if isinstance(class_index, list) and i >= len(class_index):
#                 print(f"Index {i} out of range for class_index list.")
#                 continue
#
#             img_index = torch.tensor(class_index[i]).to(device)
#             print(f"Image index tensor: {img_index}")
#
#             if self.anchor_type == "center":
#                 s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
#                 s_center_i = torch.mean(F.relu(s_anchors_i), axis=0, keepdims=True)
#                 s_anchors = s_center_i if s_anchors is None else torch.cat((s_anchors, s_center_i), axis=0)
#
#                 t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
#                 t_center_i = torch.mean(F.relu(t_anchors_i), axis=0, keepdims=True)
#                 t_anchors = t_center_i if t_anchors is None else torch.cat((t_anchors, t_center_i), axis=0)
#
#             elif self.anchor_type == "class":
#                 img_index = torch.tensor(np.random.permutation(
#                     class_index[i])[0:self.class_anchor]).to(device)
#                 s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
#                 s_anchors = s_anchors_i if s_anchors is None else torch.cat((s_anchors, s_anchors_i), axis=0)
#
#                 t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
#                 t_anchors = t_anchors_i if t_anchors is None else torch.cat((t_anchors, t_anchors_i), axis=0)
#
#         relation_loss = self.relation_loss(f_s, s_anchors, f_t, t_anchors)
#         return CCD_loss, relation_loss
#
#     def compute_contrast_idx(self, batch_label_matrix):
#         contrast_idx = torch.randint(0, batch_label_matrix.size(0), (batch_label_matrix.size(0),), device=device)
#         return contrast_idx
#
#
#
# class anchor_relation_loss(nn.Module):
#     def __init__(self, T):
#         super(anchor_relation_loss, self).__init__()
#         self.l2norm = Normalize(2).to(device)
#         self.T = T
#         self.cos = nn.CosineEmbeddingLoss().to(device)
#         self.target = torch.Tensor([1]).to(device)
#
#     def forward(self, f_s, s_anchors, f_t, t_anchors):
#         s_anchors = self.l2norm(s_anchors)
#         s_relation = torch.div(
#             torch.mm(f_s, s_anchors.clone().detach().T), self.T)
#
#         t_anchors = self.l2norm(t_anchors)
#         t_relation = torch.div(
#             torch.mm(f_t, t_anchors.clone().detach().T), self.T)
#
#         loss = uncertainity_loss(s_relation, t_relation.detach())
#         return loss
#
#
# class ContrastLoss(nn.Module):
#     def __init__(self, n_data):
#         super(ContrastLoss, self).__init__()
#         self.n_data = n_data
#
#     def forward(self, x, P):
#         bsz = x.shape[0]
#         N = x.size(1) - P
#         m = N
#         Pn = 1 / float(self.n_data)
#
#         P_pos = x.narrow(1, 0, P)
#         log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
#
#         P_neg = x.narrow(1, P, N)
#         log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
#                            P_neg.add(m * Pn + eps)).log_()
#
#         loss = - ((log_D1.squeeze().sum(0) + log_D0.view(-1,
#                                                          1).repeat(1, P).sum(0)) / bsz).sum(0) / P
#
#         return loss
#
#
# class Embed(nn.Module):
#     def __init__(self, dim_in=1024, dim_out=128):
#         super(Embed, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         self.l2norm = Normalize(2).to(device)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         x = self.l2norm(x)
#         return x
#
#
# class Normalize(nn.Module):
#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power
#
#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm)
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRALoss(nn.Module):
    def __init__(self, opt):
        super(CRALoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim).to(device)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim).to(device)
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_p, opt.nce_k, opt.nce_t, opt.nce_m).to(device)

        self.criterion_t = ContrastLoss(opt.n_data).to(device)
        self.criterion_s = ContrastLoss(opt.n_data).to(device)

        self.relation_loss = anchor_relation_loss(opt.nce_t).to(device)
        self.anchor_type = opt.anchor_type
        self.class_anchor = opt.class_anchor

        # Add l2norm attribute
        self.l2norm = Normalize(2).to(device)

    def forward(self, f_s, f_t, batch_label, class_index, num_pos, contrast_idx=None):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)

        # # Debugging `batch_label` and `class_index`
        # print(f"Batch label shape: {batch_label.shape}")
        # print(f"Class index type: {type(class_index)}")
        # print(f"Class index content: {class_index}")

        if batch_label.ndimension() > 1:
            batch_label = torch.argmax(batch_label, dim=1)

        batch_label_matrix = torch.eq(batch_label.view(-1, 1), batch_label.view(1, -1)).to(device)

        if contrast_idx is None:
            contrast_idx = self.compute_contrast_idx(batch_label_matrix)

        out_s, out_t, self.memory_s, self.memory_t = self.contrast(
            f_s, f_t, batch_label_matrix, contrast_idx)

        s_loss = self.criterion_s(out_s, num_pos)
        t_loss = self.criterion_t(out_t, num_pos)
        CCD_loss = s_loss + t_loss

        # # Debugging anchor calculations
        # print(f"Anchor type: {self.anchor_type}")
        # print(f"Class index length: {len(class_index)}")

        # Ensure anchors are initialized properly
        s_anchors = torch.empty(0, self.memory_s.size(1), device=device)
        t_anchors = torch.empty(0, self.memory_t.size(1), device=device)

        for i, class_key in enumerate(class_index):
            # Debugging individual class index
            # print(f"Processing class index {i} with key: {class_key}")

            img_index = torch.tensor(class_index[class_key], device=device)

            if self.anchor_type == "center":
                # Compute class centers for both student and teacher
                s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
                if s_anchors_i.size(0) > 0:  # Check if it's not empty
                    s_center_i = torch.mean(F.relu(s_anchors_i), axis=0, keepdims=True)
                    s_anchors = torch.cat((s_anchors, s_center_i), axis=0) if s_anchors.size(0) > 0 else s_center_i

                t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
                if t_anchors_i.size(0) > 0:  # Check if it's not empty
                    t_center_i = torch.mean(F.relu(t_anchors_i), axis=0, keepdims=True)
                    t_anchors = torch.cat((t_anchors, t_center_i), axis=0) if t_anchors.size(0) > 0 else t_center_i

            elif self.anchor_type == "class":
                img_index = torch.tensor(np.random.permutation(class_index[class_key])[0:self.class_anchor]).to(device)
                s_anchors_i = torch.index_select(self.memory_s.to(device), 0, img_index.view(-1))
                if s_anchors_i.size(0) > 0:
                    s_anchors = torch.cat((s_anchors, s_anchors_i), axis=0) if s_anchors.size(0) > 0 else s_anchors_i

                t_anchors_i = torch.index_select(self.memory_t.to(device), 0, img_index.view(-1))
                if t_anchors_i.size(0) > 0:
                    t_anchors = torch.cat((t_anchors, t_anchors_i), axis=0) if t_anchors.size(0) > 0 else t_anchors_i

        # Check if s_anchors and t_anchors are still empty
        if s_anchors.size(0) == 0 or t_anchors.size(0) == 0:
            raise ValueError("Anchor computation failed: s_anchors or t_anchors are empty.")

        # Normalize anchors and compute relation loss
        s_anchors = self.l2norm(s_anchors)
        t_anchors = self.l2norm(t_anchors)

        relation_loss = self.relation_loss(f_s, s_anchors, f_t, t_anchors)
        return CCD_loss, relation_loss

    def compute_contrast_idx(self, batch_label_matrix):
        contrast_idx = torch.randint(0, batch_label_matrix.size(0), (batch_label_matrix.size(0),), device=device)
        return contrast_idx

class anchor_relation_loss(nn.Module):
    def __init__(self, T):
        super(anchor_relation_loss, self).__init__()
        self.l2norm = Normalize(2).to(device)
        self.T = T

    def forward(self, f_s, s_anchors, f_t, t_anchors):
        s_relation = torch.div(torch.mm(f_s, s_anchors.clone().detach().T), self.T)
        t_relation = torch.div(torch.mm(f_t, t_anchors.clone().detach().T), self.T)
        loss = uncertainity_loss(s_relation, t_relation.detach())
        return loss

class ContrastLoss(nn.Module):
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x, P):
        bsz = x.shape[0]
        N = x.size(1) - P
        Pn = 1 / float(self.n_data)
        P_pos = x.narrow(1, 0, P)
        log_D1 = torch.div(P_pos, P_pos.add(N * Pn + eps)).log_()
        loss = -log_D1.sum() / bsz
        return loss

class Embed(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out

import ast
import sys
import pickle
from collections import Counter

import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from dataset import DVQA, collate_data, transform
import time
import os

# if torch.__version__ == '1.1.0':
#     from torchvision.models.resnet import resnet101 as _resnet101
# else:
from torchvision.models import resnet101 as _resnet101

# from torchvision.models import resnet152 as _resnet152

# from model import RelationNetworks

model_name = "SANVQAbeta"  # "SANVQAbeta" # "SANVQA"  # "IMGQUES"  # "IMG"  # "IMG"  # "QUES"  # "YES"
use_annotation = True if model_name == "SANDY" else False
lr = 1e-3
lr_max = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_parallel = True
lr_step = 20
lr_gamma = 2  # gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.
weight_decay = 1e-4
n_epoch = 5
reverse_question = False
batch_size = (64 if model_name == "QUES" else 32) if torch.cuda.is_available() else 4
n_workers = 0  # 4
clip_norm = 50
load_image = False


class YES(nn.Module):
    def __init__(
            self,
            n_class, yes_class_idx,

    ):
        super(YES, self).__init__()
        self.n_class = n_class
        self.yes_class_idx = yes_class_idx
        self.f = nn.Sequential(
            nn.Linear(10, 3),
        )

    def forward(self, image, question, question_len):
        result = np.zeros([image.shape[0], self.n_class])
        result[:, self.yes_class_idx] = 1
        return torch.Tensor(result)


class IMG(nn.Module):
    def __init__(
            self,
            n_class, encoded_image_size=7, conv_output_size=2048, mlp_hidden_size=1024, dropout_rate=0.5

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(IMG, self).__init__()
        self.n_class = n_class

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet101(
        #     pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        resnet = _resnet101(pretrained=True)
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(
            self.dropout,
            nn.Linear(conv_output_size * encoded_image_size * encoded_image_size, mlp_hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(mlp_hidden_size, self.n_class))

        # self.mlp = nn.Sequential(self.dropout,
        #                          nn.Linear(conv_output_size, mlp_hidden_size),
        #                          nn.ReLU(),
        #                          self.dropout,
        #                          nn.Linear(mlp_hidden_size, self.n_class))  # including AvgPool2d, 051019 afternoon by Xin

        self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        final_out = self.mlp(conv_out.view(conv_out.size(0), -1).contiguous())

        return final_out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
            # for c in list(self.resnet.children())[6:]:
            #     for p in c.parameters():
            #         p.requires_grad = fine_tune
        else:
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4.
            # Before that not-fine-tuned are block 1 and preliminary blocks.
            # >>> print(len(list(resnet.children()))) >>> 10 -> we only utilize up to the 8th block.
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True

        # # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        # for c in list(self.resnet.children())[5:]:
        #     for p in c.parameters():
        #         p.requires_grad = fine_tune


class IMGQUES(nn.Module):
    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(IMGQUES, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(
            pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * encoded_image_size * encoded_image_size + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        # print(embed_pack)
        output, (h, c) = self.lstm(embed_pack)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        last_output = output.gather(
            time_dimension, idx).squeeze(time_dimension)

        conv_lstm_feature = torch.cat((conv_out, last_output), 1)

        return self.mlp(conv_lstm_feature)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
            # for c in list(self.resnet.children())[6:]:
            #     for p in c.parameters():
            #         p.requires_grad = fine_tune
        else:
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4.
            # Before that not-fine-tuned are block 1 and preliminary blocks.
            # >>> print(len(list(resnet.children()))) >>> 10 -> we only utilize up to the 8th block.
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True


class Attention(nn.Module):  # SANVQAbeta
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.5):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


# class Attention(nn.Module):  # SANVQAtri
#     def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.5):
#         super(Attention, self).__init__()
#
#         # self.v_conv = nn.Conv2d(v_features, mid_features, 1)  # let self.lin take care of bias
#         # self.q_lin = nn.Linear(q_features, mid_features)
#         # self.x_conv = nn.Conv2d(mid_features, glimpses, 1)
#         #
#         # self.drop = nn.Dropout(drop)
#         # self.relu = nn.ReLU(inplace=True)
#
#         self.w_i_a = nn.mlp
#
#     def forward(self, v, q):
#         v = v.view(v.size(0), v.size(1), -1).contiguous().permute(0,2,1)
#
#         v = self.v_conv(self.drop(v))
#         q = self.q_lin(self.drop(q))
#         q = tile_2d_over_nd(q, v)
#         x = self.relu(v + q)
#         x = self.x_conv(self.drop(x))
#         return x

# def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
#     super(Attention, self).__init__()
#
#     # self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
#     # self.q_lin = nn.Linear(q_features, mid_features)
#
#     self.v_q_conv_1 = nn.Conv2d(v_features + q_features, mid_features, 1, bias=False)
#     self.v_q_conv_2 = nn.Conv2d(mid_features, glimpses, 1)
#     self.drop = nn.Dropout(drop)
#     self.relu = nn.ReLU(inplace=True)
#
# def forward(self, v, q):
#     tiled_lstm_final_output = tile_2d_over_nd(q, v)
#
#     concat_v_q = torch.cat((tiled_lstm_final_output, v), dim=1)
#     attention_mid = self.relu(self.v_q_conv_1(concat_v_q))
#     attention_final = self.relu(self.v_q_conv_2(attention_mid))
#     return attention_final

# v = self.v_conv(self.drop(v))
# q = self.q_lin(self.drop(q))
# q = tile_2d_over_nd(q, v)
# x = self.relu(v + q)
# x = self.x_conv(self.drop(x))
# return x


def apply_attention(input, attention):  # softmax weight, then weighted average
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)  # = 2

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1)  # [n, 1, c, spatial]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, glimpses, 1, spatial]
    weighted = attention * input  # [n, glimpses, channel, spatial]
    weighted_mean = weighted.sum(dim=-1)  # [n, glimpses, channel]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    # input(feature_map.shape)
    # input(feature_vector.shape)
    spatial_size = feature_map.dim() - 2
    # tiled = feature_vector.view(n, c, *([1] * spatial_size)).repeat(1, int(feature_map.size()[1]/c), 1, 1).expand_as(
    #     feature_map)
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).repeat(1, 1, feature_map.size()[-2],
                                                                    feature_map.size()[-1])
    return tiled


class SANVQA(nn.Module):
    '''
    We implement SANVQA based on https://github.com/Cyanogenoid/pytorch-vqa.
    A SAN implementation for show, ask, attend and tell

    Currently as one-hop
    TODO: change to two-hops
    '''

    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5, glimpses=2

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(SANVQA, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet152(
        #     pretrained=True)
        resnet = torchvision.models.resnet101(
            pretrained=True)  # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling
        modules = list(resnet.children())[:-2]
        # modules = list(resnet.children())[:-1]  # including AvgPool2d, 051019 afternoon by Xin

        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.attention = Attention(conv_output_size, lstm_hidden, mid_features=512, glimpses=glimpses, drop=0.5)

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * glimpses + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        self.fine_tune()  # define which parameter sets are to be fine-tuned
        self.hop = 1

    def forward(self, image, question, question_len):  # this is an image blind example (as in section 4.1)
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)

        # normalize by feature map, why need it and why not??
        # conv_out = conv_out / (conv_out.norm(p=2, dim=1, keepdim=True).expand_as(
        #     conv_out) + 1e-8)  # Section 3.1 of show, ask, attend, tell

        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        # conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        lstm_output, (h, c) = self.lstm(embed_pack)

        # pad packed sequence to get last timestamp of lstm hidden
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), lstm_output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        lstm_final_output = lstm_output.gather(
            time_dimension, idx).squeeze(time_dimension)

        attention = self.attention(conv_out, lstm_final_output)
        weighted_conv_out = apply_attention(conv_out,
                                            attention)  # (n, glimpses * channel) ## Or should be (n, glimpses * channel, H, W)?
        # augmented_lstm_output = (weighted_conv_out + lstm_final_output)
        augmented_lstm_output = torch.cat((weighted_conv_out, lstm_final_output), 1)

        if self.hop == 2:
            raise NotImplementedError
            # attention = self.attention(conv_out, lstm_final_output)
            # weighted_conv_out = apply_attention(conv_out, attention)
            # augmented_lstm_output = (weighted_conv_out + lstm_final_output)

        return self.mlp(augmented_lstm_output)

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
        else:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune

        for p in self.mlp.parameters():
            p.requires_grad = True

        for p in self.attention.parameters():
            print(p.requires_grad, "p.requires_grad")
            p.requires_grad = True


class QUES(nn.Module):

    def __init__(
            self, n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, dropout_rate=0.5, mlp_hidden_size=1024):
        super(QUES, self).__init__()
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        # self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(lstm_hidden, mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        # self.fine_tune()  # define which parameter sets are to be fine-tuned

    def forward(self, image, question, question_len):
        #  https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        # print(embed_pack)
        output, (h, c) = self.lstm(embed_pack)
        # h_tile = h.permute(1, 0, 2).expand(
        #     batch_size, n_pair * n_pair, self.lstm_hidden
        # )

        # _, (h, c) = self.lstm(question)

        # input(h.shape)
        # out = self.mlp(h.squeeze(0))

        # Extract the outputs for the last timestep of each example

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)
        # if output.is_cuda:
        #     idx = idx.cuda(output.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = output.gather(
            time_dimension, idx).squeeze(time_dimension)
        # input(last_output.shape)
        return self.mlp(last_output)

        # emb_len = np.array(question_len)
        # sorted_idx = np.argsort(-emb_len)
        # embed = embed[sorted_idx]
        # emb_len = emb_len[sorted_idx]
        # unsorted_idx = np.argsort(sorted_idx)
        #
        # packed_emb = torch.nn.utils.rnn.pack_padded_sequence(embed, emb_len, batch_first=True)
        # output, hn = self.rnn(packed_emb)
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        # unpacked = unpacked.transpose(0, 1)
        # unpacked = unpacked[torch.LongTensor(unsorted_idx)]
        # return unpacked
        #
        #
        # return out


def train_YES(epoch):  # basically trim over loss calculation and back prop for YES model calculation
    raise NotImplementedError


def valid_YES(epoch, val_split="val_easy",
              load_image=True):  # basically trimmed version of train() and valid(), except without image_loading and GPU tensor operation
    raise NotImplementedError


def train(epoch, load_image=True, model_name=None):
    # train_set = DataLoader(
    #     DVQA(
    #         sys.argv[1],
    #         transform=transform,
    #         reverse_question=reverse_question,
    #         use_preprocessed=True,
    #         load_image=load_image,
    #         load_from_hdf5=load_from_hdf5
    #
    #     ),
    #     batch_size=batch_size,
    #     num_workers=n_workers,
    #     shuffle=True,
    #     collate_fn=collate_data,
    # )

    model.train(True)  # train mode

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0  # it will change when loop over data
    # print("Using moving_Loss? ", moving_loss == 0)

    # start = time.time()
    # print("start reading data")

    print(device)
    print(next(model.parameters()).is_cuda, "next(model.parameters()).is_cuda")
    for i, (image, question, q_len, answer, question_class) in enumerate(pbar):
        # end = time.time()
        # print("start another round of data", end - start)
        # start = time.time()
        # input(image.shape)
        image, question, q_len, answer = (
            image.to(device),
            question.to(device),
            torch.tensor(q_len),
            answer.to(device),
        )
        # print(image.shape)
        # print(question.shape)

        # end = time.time()
        # print("finished to(device)", end - start)
        # start = time.time()

        model.zero_grad()
        output = model(image, question, q_len)

        loss = criterion(output, answer)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        # print(output.data.cpu().numpy().argmax(1))
        # print(answer.data.cpu().numpy())

        # end = time.time()
        # print("finished loss prop", end - start)
        # start = time.time()

        correct = output.data.cpu().numpy().argmax(1) == answer.data.cpu().numpy()
        correct = correct.sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct
            # print("moving_loss = correct")

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01
            # print("moving_loss = moving_loss * 0.99 + correct * 0.01")

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}; Correct:{:.5f}; LR: {:.6f}'.format(
                epoch + 1,
                loss.detach().item(),  # 0.00 for YES model
                moving_loss,
                correct,
                optimizer.param_groups[0]['lr'],  # 0.00  for YES model
            )
        )
        if (("IMG" in model_name) or ("SAN" in model_name)) and i % 10000 == 0 and i != 0:
            # valid(epoch + float(i * batch_size / 2325316), model_name=model_name, val_split="val_easy",
            #       load_image=load_image)
            valid(epoch + float(i * batch_size / 2325316), valid_set_easy, model_name=model_name,
                  load_image=load_image, val_split="val_easy")

            model.train(True)

        # end = time.time()
        # print("finished accuracy calculation", end - start)
        # start = time.time()


def valid(epoch, valid_set, load_image=True, model_name=None, val_split="val_easy"):
    print("Inside validation ", epoch)
    # valid_set = DataLoader(
    #     DVQA(
    #         sys.argv[1],
    #         val_split,
    #         transform=None,
    #         reverse_question=reverse_question,
    #         use_preprocessed=True,
    #         load_image=load_image,
    #         load_from_hdf5=load_from_hdf5
    #     ),
    #     batch_size=batch_size // 2,
    #     num_workers=4,
    #     collate_fn=collate_data,  ## shuffle=False
    #
    # )
    # valid_set=valid_set_easy if val_split=="val_easy" else val
    dataset = iter(valid_set)

    model.eval()  # eval_mode
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for i, (image, question, q_len, answer, answer_class) in enumerate(tqdm(dataset)):
            image, question, q_len = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
            )

            output = model(image, question, q_len)
            correct = output.data.cpu().numpy().argmax(1) == answer.numpy()
            for c, class_ in zip(correct, answer_class):
                if c:  # if correct
                    class_correct[class_] += 1
                class_total[class_] += 1

            if (("IMG" in model_name) or ("SAN" in model_name)) and type(epoch) == type(0.1) and (
                    i * batch_size // 2) > (
                    6e4):  # intermediate train, only val on 10% of the validation set
                break  # early break validation loop

    class_correct['total'] = sum(class_correct.values())
    class_total['total'] = sum(class_total.values())

    print("class_correct", class_correct)
    print("class_total", class_total)

    with open('log/log_' + model_name + '_{}_'.format(round(epoch + 1, 4)) + val_split + '.txt', 'w') as w:
        for k, v in class_total.items():
            w.write('{}: {:.5f}\n'.format(k, class_correct[k] / v))
        # TODO: save the model here!

    print('Avg Acc: {:.5f}'.format(class_correct['total'] / class_total['total']))


if __name__ == '__main__':
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])
    global load_from_hdf5
    load_from_hdf5 = ast.literal_eval(sys.argv[2])
    print("load_from_hdf5", load_from_hdf5)
    print(n_answers, "n_answers")

    if model_name == "YES":
        yes_answer_idx = dic['answer_dic']["yes"]
        model = YES(n_answers, yes_answer_idx)
        load_image = False

    elif model_name == "IMG":
        # load_from_hdf5 = False
        model = IMG(n_answers, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        # if data_parallel:
        #     model = nn.DataParallel(model)
        load_image = True
        model = model.to(device)
    elif model_name == "QUES":
        load_from_hdf5 = False
        load_image = False
        model = QUES(n_answers, n_vocab=n_words)
        model = model.to(device)

    elif model_name == "IMGQUES":
        model = IMGQUES(n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        load_image = True
        model = model.to(device)

    elif model_name == "SANVQA" or "SANVQAbeta":
        model = SANVQA(n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        load_image = True
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma) # Decays the learning rate of each parameter group by gamma every step_size epochs.

    train_set = DataLoader(
        DVQA(
            sys.argv[1],
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5

        ),
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        collate_fn=collate_data,
    )

    valid_set_easy = DataLoader(
        DVQA(
            sys.argv[1],
            "val_easy",
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5
        ),
        batch_size=batch_size // 2,
        num_workers=n_workers,
        collate_fn=collate_data,  ## shuffle=False

    )

    valid_set_hard = DataLoader(
        DVQA(
            sys.argv[1],
            "val_hard",
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5
        ),
        batch_size=batch_size // 2,
        num_workers=n_workers,
        collate_fn=collate_data,  ## shuffle=False

    )

    for epoch in range(n_epoch):
        # if scheduler.get_lr()[0] < lr_max:
        #     scheduler.step()
        print("epoch=", epoch)

        # TODO: add load model from checkpoint
        checkpoint_name = 'checkpoint/checkpoint_' + model_name + '_{}.model'.format(str(epoch + 1).zfill(3))
        # if os.path.exists(checkpoint_name):
        #     # model.load_state_dict(torch.load(model.state_dict())
        #     model.load_state_dict(
        #         torch.load(checkpoint_name, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        #     continue

        train(epoch, load_image=load_image, model_name=model_name)
        valid(epoch, valid_set_easy, model_name=model_name, load_image=load_image, val_split="val_easy")
        valid(epoch, valid_set_hard, model_name=model_name, load_image=load_image, val_split="val_hard")
        with open(checkpoint_name, 'wb'
                  ) as f:
            torch.save(model.state_dict(), f)

        print("model saved! epoch=", epoch)

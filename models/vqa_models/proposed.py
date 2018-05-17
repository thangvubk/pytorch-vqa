import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import config


class Proposed(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Proposed, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.fusion = Fusion(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            drop=0.5,
        )

        self.fusion1 = Fusion(
            v_features=vision_features,
            q_features=512,
            mid_features=512,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=512 + 1024,  # equals fusion mid feature*2
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        self.c_att = Channel_Attention_Layer(512)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        #i#iprint(v.size(), q.size())
        f = self.fusion(v, q)
        f1 = self.fusion1(v, f)
        #att=self.c_att(f)
        #q = q*att

        #v = apply_attention(v, a)
        #print(f.size(), q.size())

        combined = torch.cat([f, q], dim=1)
        answer = self.classifier(combined)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)

class Channel_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention_Layer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel*2),
                nn.Sigmoid()
        )

    def forward(self, x):
        #b, c, _, _ = x.size()
        #x = self.avg_pool(x).view(b, c)
        #x = self.fc(x).view(b, c, 1, 1)
        #y = self.fc(x)
        return self.fc(x)

class Spatial_Attention_Layer(nn.Module):
    def __init__(self, channel):
        super(Spatial_Attention_Layer, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        b, _, h, w = x.size()
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(x)
        x = x.view(b, 1, h, w)
        return x

class Fusion(nn.Module):
    def __init__(self, v_features, q_features, mid_features, drop=0.0):
        super(Fusion, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, padding=0)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_features, 512, 1, padding=0),
            #nn.Sigmoid()
            #nn.ReLU(inplace=True),
            #nn.AdaptiveAvgPool2d(1)
            #nn.Dropout(drop, inplace=True)
            #nn.Conv2d(mid_features*2, mid_features*2, 3, padding=1),
            #nn.ReLU(inplace=True)
        )


        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.c_att = Channel_Attention_Layer(mid_features)
        self.s_att = Spatial_Attention_Layer(mid_features)
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, v, q):
        #print(v.size(), q.size())
        v1 = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v1)
        x = self.relu(v1+q)#torch.cat([v1, q], 1)
        
        #x = self.fusion(self.drop(x))
        #x = x.view(x.size(0), -1)
        y1 = self.s_att(self.drop(x))
        #y2 = self.c_att(self.drop(x))
        x = (y1 + 0)*v
        x = x.sum(dim=3).sum(dim=2)
        #x = self.pooling(x)
        x = x.view(x.size(0), -1)
        #x = self.attention(x)
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

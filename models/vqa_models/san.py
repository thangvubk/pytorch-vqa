import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init 
from torch.nn.utils.rnn import pack_padded_sequence
class SAN(nn.Module):
    def __init__(self, vocab_size, drop=0.6):
        super(SAN, self).__init__()
        self.image_module = ImageEmbedding(drop=drop)
        self.question_module = QuestionEmbedding(vocab_size, dropout=drop)
        self.attention_module = Attention(drop_ratio=drop)
    
    def forward(self, image, ques, ques_len):
        embed_image = self.image_module(image)
        embed_quest = self.question_module(ques, ques_len)#list(ques_len.data))
        return self.attention_module(embed_quest, embed_image)

class Attention(nn.Module): # Extend PyTorch's Module class
    def __init__(self, input_size=1024, att_size=512, img_seq_size=196, output_size=1000, drop_ratio=0.8):
        super(Attention, self).__init__() # Must call super __init__()
        self.input_size = input_size
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio

        self.tan = nn.Tanh()
        self.dp = nn.Dropout(drop_ratio)
        self.sf = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(input_size, att_size, bias=True)
        self.fc12 = nn.Linear(input_size, att_size, bias=False)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        self.fc21 = nn.Linear(input_size, att_size, bias=True)
        self.fc22 = nn.Linear(input_size, att_size, bias=False)
        self.fc23 = nn.Linear(att_size, 1, bias=True)

        self.fc = nn.Linear(input_size, output_size, bias=True)

        # d = input_size | m = img_seq_size | k = att_size
    def forward(self, ques_feat, img_feat):  # ques_feat -- [batch, d] | img_feat -- [batch_size, m, d]
        #  print(img_feat.size(), ques_feat.size())
        B = ques_feat.size(0)

        # Stack 1
        ques_emb_1 = self.fc11(ques_feat)  # [batch_size, att_size]
        img_emb_1 = self.fc12(img_feat)

        h1 = self.tan(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)

        h1_emb = self.fc13(self.dp(h1))
        #h1_emb = self.dp(self.fc13(h1))
        #h1_emb = self.fc13(h1)
        p1 = self.sf(h1_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att1 = p1.matmul(img_feat)
        u1 = ques_feat + img_att1.view(-1, self.input_size)

        # Stack 2
        ques_emb_2 = self.fc21(u1)  # [batch_size, att_size]
        img_emb_2 = self.fc22(img_feat)

        h2 = self.tan(ques_emb_2.view(B, 1, self.att_size) + img_emb_2)

        h2_emb = self.fc23(self.dp(h2))
        #h2_emb = self.dp(self.fc23(h2))
        #h2_emb = self.fc23(h2)
        p2 = self.sf(h2_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att2 = p2.matmul(img_feat)
        u2 = u1 + img_att2.view(-1, self.input_size)

        # score
        score = self.fc(u2)

        return score

class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size=1024, feature_type='VGG', drop=0.8):
        super(ImageEmbedding, self).__init__() # Must call super __init__()

        if feature_type == 'VGG':
            self.img_features = 512
        elif feature_type == 'Residual':
            self.img_features = 2048
        else:
            print('Unsupported feature type: \'{}\''.format(feature_type))
            return None

        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.img_features, self.hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(drop)

    def forward(self, input):
        input = input / (input.norm(p=2, dim=1, keepdim=True).expand_as(input) + 1e-8)
        # input: [batch_size, 14, 14, 512]
        input = input.permute(0, 2, 3, 1).contiguous()

        intermed = self.linear(input.view(-1,self.img_features)).view(
                                    -1, 196, self.hidden_size)
        return self.dropout(self.tanh(intermed))
        #return self.tanh(intermed)

class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size=500, hidden_size=1024, rnn_size=1024, num_layers=2, dropout=0.8, seq_length=26, use_gpu=True):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()

	self.use_gpu = use_gpu
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=True,
                batch_first=True, dropout=dropout)

        return

    def forward(self, ques_vec, ques_len):            # forward(self, ques_vec, ques_len) | ques_vec: [batch_size, 26]
        B, W = ques_vec.size()

        # Add 1 to vocab_size, since word idx from 0 to vocab_size inclusive
        one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2,
                        ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)

        # To remove additional column in one_hot, use slicing
        one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            one_hot_vec = one_hot_vec.cuda()

        x = self.lookuptable(one_hot_vec)

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.dropout(self.tanh(x))
        #emb_vec = self.tanh(x)

        # h: [batch_size or B, 26 or W, hidden_size]
        h, _ = self.LSTM(emb_vec)

        x = ques_len.data.cpu() - 1
        mask = torch.zeros(B, W).scatter_(1, x.view(-1, 1), 1)
        mask = Variable(mask.view(B, W, 1), requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            mask = mask.cuda()

        h = h.transpose(1,2)
        # print(h.size(), mask.size())

        # output: [B, hidden_size]
        return torch.bmm(h, mask).view(B, -1)

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


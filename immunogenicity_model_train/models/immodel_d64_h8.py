import torch
import torch.nn as nn
import math
import numpy as np

vocab = {'A': 1, 'W': 2, 'V': 3, 'C': 4, 'H': 5, 'T': 6, 'E': 7, 'K': 8, 'N': 9, 'P': 10, 
         'I': 11, 'L': 12, 'S': 13, 'D': 14, 'G': 15, 'Q': 16, 'R': 17, 'Y': 18, 'F': 19, 'M': 20, '-': 0}

vocab_size = len(vocab)

# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
tgt_len = 14+34 # concat最大长度

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn   

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
                
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        
        
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.hla2pep_attn = MultiHeadAttention()
        self.pep2hla_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, hla_enc_outputs, pep_enc_outputs, hla2pep_attn_mask, pep2hla_attn_mask):
        
        hla2pep_context, hla2pep_attn = self.hla2pep_attn(hla_enc_outputs, pep_enc_outputs, pep_enc_outputs, hla2pep_attn_mask)
        pep2hla_context, pep2hla_attn = self.pep2hla_attn(pep_enc_outputs, hla_enc_outputs, hla_enc_outputs, pep2hla_attn_mask)
        
        hla2pep_context = self.pos_ffn(hla2pep_context) 
        pep2hla_context = self.pos_ffn(pep2hla_context) 
        
        return hla2pep_context, pep2hla_context, hla2pep_attn, pep2hla_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoderlayer = DecoderLayer() 

    def forward(self, hla_inputs, pep_inputs, hla_enc_outputs, pep_enc_outputs): 

        hla2pep_attn_mask = get_attn_pad_mask(hla_inputs, pep_inputs) 
        pep2hla_attn_mask = get_attn_pad_mask(pep_inputs, hla_inputs)

        hla2pep_attns, pep2hla_attns = [], []            
        hla2pep_context, pep2hla_context, hla2pep_attn, pep2hla_attn = self.decoderlayer(hla_enc_outputs, pep_enc_outputs, hla2pep_attn_mask, pep2hla_attn_mask)
            
        hla2pep_attns.append(hla2pep_attn)
        pep2hla_attns.append(pep2hla_attn)
            
        return hla2pep_context, pep2hla_context, hla2pep_attns, pep2hla_attns    
     
class IM_TransferModel(nn.Module):
    def __init__(self):
        super(IM_TransferModel, self).__init__()
               
        self.hla_encoder = Encoder().to(device)
        self.pep_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)        
        
        # prediction layers for EL
        self.fc1 = nn.Linear(tgt_len * d_model + 4096*2, 4096).to(device)
        self.bn1 = nn.BatchNorm1d(4096).to(device)
        self.re1 = nn.ReLU().to(device)
        self.dp1 = nn.Dropout(p=0.2).to(device)
        
        
        self.fc2 = nn.Linear(4096, 4096).to(device)
        self.bn2 = nn.BatchNorm1d(4096).to(device)
        self.re2 = nn.ReLU().to(device)
        
        self.predict = nn.Linear(4096, 1).to(device)        
   
    def forward(self, hla_inputs, pep_inputs, ba, el):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''

        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)

        
        hla2pep_context, pep2hla_context, hla2pep_attns, pep2hla_attns = self.decoder(hla_inputs, pep_inputs, hla_enc_outputs, pep_enc_outputs)
        
        outputs = torch.cat((hla2pep_context, pep2hla_context), 1) # concat hla & pep embedding
        
        outputs = outputs.view(outputs.shape[0], -1) 
        
        x_fusion = torch.cat([outputs,ba,el],dim=1)
        
        # Predict
        x_fc1 = self.fc1(x_fusion)
        x_bn1 = self.bn1(x_fc1)
        x_re1 = self.re1(x_bn1)
        x_dp1 = self.dp1(x_re1)
        
        
        x_fc2 = self.fc2(x_dp1)
        x_bn2 = self.bn2(x_fc2)
        x_re2 = self.re2(x_bn2)
                
        y_im = self.predict(x_re2)
                        
        return y_im, hla2pep_attns, pep2hla_attns, hla_enc_self_attns, pep_enc_self_attns, x_fc1, x_bn1, x_re1, x_dp1, x_fc2, x_bn2, x_re2, outputs
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv
from itertools import groupby
import pywt
from testmamba import ModelArgs, Mamba



class Personal_Geo_GCN(nn.Module):
    def __init__(self, poi_dim, geoGCN_num,device):
        super(Personal_Geo_GCN, self).__init__()
        # self.conv1 = GCNConv(poi_dim, poi_dim)
        # self.conv2 = GCNConv(poi_dim, poi_dim)
        self.device=device
        self.act = nn.LeakyReLU()
        self.geo_convs = nn.ModuleList()
        for _ in range(geoGCN_num): # 2
            self.geo_convs.append(GCNConv(poi_dim,poi_dim)) #(64,64)

    def forward(self, poi_embeddings, distance_matrix, seqInDay ,GCN_layer_num,sampleIdOfDay_set): #[1431, 64], [1430, 1430], [80, 15], 2 
        poi_emb=poi_embeddings.weight[1:,:] #[1430, 64]
        
        # Pair sampleIdOfDay_set with seqInDay
        paired = zip(sampleIdOfDay_set, seqInDay)

        # Sort by sampleIdOfDay_set
        paired_sorted = sorted(paired, key=lambda x: x[0])

        # Group by sampleIdOfDay_set
        grouped = groupby(paired_sorted, key=lambda x: x[0])

        # Collect seqInDay for each group into a list
        samples = [torch.stack([x[1] for x in group]) for _, group in grouped]

        seq_GCN_embeddings = []
        # Define the distance intervals
        intervals = (torch.arange(0, distance_matrix.max() + 0.5, 0.5)).to(self.device)

        for sample in samples:
            # Remove padding zeros
            lastDay = sample[-1][sample[-1] != 0]

            # Calculate the distance spans for each transition
            spans = distance_matrix[lastDay[:-1] - 1, lastDay[1:] - 1]

            # Determine the interval for each span
            span_intervals = torch.bucketize(spans, intervals)

            # Calculate the weight for each interval
            weights = torch.bincount(span_intervals, minlength=len(intervals)).float()
            weights /= weights.sum()

            # Create a new weight matrix for this day
            weight_matrix = torch.zeros_like(distance_matrix)

            for id,i in enumerate(torch.unique(span_intervals)):
                mask = (distance_matrix >= intervals[i-1]) & (distance_matrix < intervals[i]) #在该距离跨度内的poi对
                weight_matrix[mask] = weights[i]
            weight_matrix.fill_diagonal_(1) #对角线为1

            #GCN前进行规范化,(归一化)
            adj_dig = torch.clamp(torch.pow(torch.sum(weight_matrix, dim=-1, keepdim=True),0.5), min=1e-12) #[1430, 1]
            update_adj = weight_matrix/adj_dig/adj_dig.transpose(-1,-2)#[1430,1430]
            update_adj=update_adj.to(self.device)

            #进行GCN
            x_fin = [poi_emb]
            layer = poi_emb
            for f in range(GCN_layer_num): #公式6  每天的seq的embedding 用"seq距离跨度的权重矩阵"进行更新
                layer = torch.matmul(update_adj,layer) #[1430, 64]
                layer = torch.tanh(layer) #[1430, 64]
                x_fin += [layer] #[3,[1430, 64]] [raw,l1,l2]
            x_fin = torch.stack(x_fin,dim=1) #[1430, 3, 64]
            out = torch.sum(x_fin,dim=1) #[1430, 64] GCN完毕的全部poi_embedding
            GCN_poi_emb = torch.cat([poi_embeddings.weight[0, :].unsqueeze(dim=0),out], dim=0) #[1431, 64] 把去掉的第一行 又放回来了
            # sample=sample.to('cpu')

            sample_emb = GCN_poi_emb[sample,:] #[15, 64]<-[15,]
            sample_emb *= sample_emb.shape[1] ** 0.5 #  *8   #[15, 64]

            seq_GCN_embeddings.append(sample_emb) 

        seq_GCN_embeddings = torch.cat(seq_GCN_embeddings, dim=0)

        return seq_GCN_embeddings




class Embedding_Layer(nn.Module):
    def __init__(self, args):
        super(Embedding_Layer, self).__init__()
        # self.poi_embedding = poi_embedding #[1431,64]
        self.hour_embedding = nn.Embedding(args.hour_num+1, args.hour_dim, padding_idx=0) #[25,32]
        self.isweekend_embedding = nn.Embedding(args.isweekend_num+1, args.isweekend_dim, padding_idx=0)#[3,32]
        self.user_embedding = nn.Embedding(args.user_num+1, args.user_dim, padding_idx=0)#[169,32]
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, seq_poi_embeddings, hour_set,isweekend_set,user_set):
        poi_embeddings = seq_poi_embeddings #[157,15,64]
        hour_embeddings = self.hour_embedding(hour_set) #[157,15,32]
        isweenkend_embeddings = self.isweekend_embedding(isweekend_set) #[157,15,32]
        user_embeddings = self.user_embedding(user_set) #[157,15,32]
        seq_embeddings=torch.cat((poi_embeddings,hour_embeddings,isweenkend_embeddings,user_embeddings),dim=2) #[157,15,128]

        return seq_embeddings #[157,15,128]


class Distance_Time_Span_Embedding_Layer(nn.Module):
    def __init__(self, args):
        super(Distance_Time_Span_Embedding_Layer, self).__init__()
        self.distance_embedding = nn.Embedding(257, args.hidden_dim, padding_idx=0)  # 256 unique values + 1 for padding
        self.time_embedding = nn.Embedding(25, args.hidden_dim, padding_idx=0)  # 24 unique values + 1 for padding
        self.dis_dropout = nn.Dropout(args.dropout_rate)
        self.time_dropout = nn.Dropout(args.dropout_rate)

    def forward(self, distance_span_set, time_span_set): #[80, 15] , [80, 15]
        distance_span_embedding = self.distance_embedding(distance_span_set) #[80, 15, 128]
        time_span_embedding = self.time_embedding(time_span_set)
        distance_span_embedding = self.dis_dropout(distance_span_embedding)
        time_span_embedding = self.time_dropout(time_span_embedding)

        return distance_span_embedding, time_span_embedding



class SelfAttention_Layer(nn.Module):
    def __init__(self, hidden_dim,dropout):
        super(SelfAttention_Layer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim,  2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_embeddings, mask): #[80, 15, 128], [80, 15]
        Q = self.query(seq_embeddings)
        K = self.key(seq_embeddings)
        V = self.value(seq_embeddings)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1)) #[80, 15, 15]
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)
        output = output.masked_fill(mask.unsqueeze(-1) == 0, 0)

        #正则化&drop
        # x = self.dropout(self.norm1(output + Q)) 
        # forward = self.feed_forward(x) 
        # output = self.dropout(self.norm2(forward + x)) 
        return output
     


class SpanLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpanLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, seq_emb, dist_emb, time_emb, hx=None): #[80, 128],[80, 128],[80, 128]
        if hx == (None,None):
            hx = seq_emb.new_zeros(seq_emb.size(0), self.hidden_dim), \
                 seq_emb.new_zeros(seq_emb.size(0), self.hidden_dim)

        h, c = hx  #[80, 128], [80, 128]

        # [80, 512]
        gates = (F.linear(seq_emb, self.weight_ih, self.bias) +
                 F.linear(dist_emb, self.weight_ih) +
                 F.linear(time_emb, self.weight_ih) +
                 F.linear(h, self.weight_hh))

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = (forget_gate * c) + (input_gate * cell_gate)
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next #[80, 128], [80, 128]

class Span_LSTM_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Span_LSTM_Layer, self).__init__()
        self.lstm_cell = SpanLSTMCell(input_dim, hidden_dim)

    def forward(self, seq_emb, dist_emb, time_emb, mask): #[80, 15, 128], [80, 15, 128], [80, 15, 128], [80, 15]
        batch_size, seq_length, _ = seq_emb.size()
        h, c = None, None
 
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, seq_length, self.lstm_cell.hidden_dim).to(seq_emb.device)

        # Process each element in the sequence one at a time
        for t in range(seq_length):
            h, c = self.lstm_cell(seq_emb[:, t, :], dist_emb[:, t, :], time_emb[:, t, :], (h, c))
            outputs[:, t, :] = h  #[80, 15, 128]

        # Get the last valid output of each sequence
        lengths = mask.sum(dim=1)
        dayOutputs = torch.stack([outputs[i, length-1] for i, length in enumerate(lengths)])

        return outputs, dayOutputs #[80, 15, 128], [80, 128]


class LSTM_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(LSTM_Layer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,dropout=0)

    def forward(self, inputs, mask): #[157, 15, 128], [157, 15]
        # Compute the length of each sequence in the batch
        lengths = mask.sum(dim=1) #[157,]

        # Sort by length to use pack_padded_sequence
        lengths, sort_idx = lengths.sort(0, descending=True) #[157,], [157,]
        inputs = inputs[sort_idx] #[157, 15, 128]

        # Pack the sequence
        packed_inputs = pack_padded_sequence(inputs, lengths.cpu().numpy(), batch_first=True)

        # Pass through LSTM
        packed_outputs, _ = self.lstm(packed_inputs)

        # Unpack the sequence
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # Unsort the outputs
        _, unsort_idx = sort_idx.sort(0)      
        outputs = outputs[unsort_idx] #[157, 15, 128]

        # Get the last valid output of each sequence
        unsorted_lengths = mask.sum(dim=1)
        dayOutputs = torch.stack([outputs[i, length-1] for i, length in enumerate(unsorted_lengths)])

        return outputs,dayOutputs #[157, 128]
    


class SeqMamba_Module(nn.Module):
    def __init__(self, mamba_args, L=1):
        super(SeqMamba_Module, self).__init__()
        # Initialize MambaBlock and PFFN here
        self.MambaBlock = Mamba(mamba_args)
        self.PFFN = nn.Sequential(
            nn.Linear(mamba_args.d_model, mamba_args.d_model * 4),  # First dense layer
            nn.GELU(),  # GELU activation
            nn.Linear(mamba_args.d_model * 4, mamba_args.d_model)  # Second dense layer
        )
        self.L = L  # Number of layers

    def forward(self, seq_embeddings):
        # Loop for L layers
        for _ in range(self.L):
            # Pass through MambaBlock
            mamba_output = self.MambaBlock(seq_embeddings)
            
            # Pass through PFFN
            pffn_output = self.PFFN(mamba_output)
            
            # Combine Mamba output and PFFN output
            combined_output = mamba_output + pffn_output
            
            # Apply layer normalization
            seq_embeddings = nn.LayerNorm(combined_output.size()[1:])(combined_output)

        return seq_embeddings  # Return the final output after L layers


class DFT_Layer(nn.Module):
    def __init__(self):
        super(DFT_Layer, self).__init__()
        # Initialize your layers here

    def forward(self, seqs,     seq_embeddings):

        transformed_seq_embeddings=seq_embeddings
        aggregated_embeddings = []

        # For each sample
        for i in range(seqs.shape[0]):
            
            seqMask=[seqs[i] != 0]
            real_poi = seqs[i][seqMask]
            # Create a binary sequence indicating whether the last POI was visited at each time point
            binary_seq = (real_poi == real_poi[-1]).float()

            # Compute the FFT of the binary sequence and normalize it to probabilities
            fft_prob = torch.abs(torch.fft.fft(binary_seq))
            fft_prob[-1] = 0
            fft_prob = fft_prob / fft_prob.sum()

            # Shift the probabilities to the left by one position
            fft_prob = torch.roll(fft_prob, shifts=1)#[46,]个概率

            padding_size = seqs.shape[1] - fft_prob.shape[0]  # Calculate the padding size
            fft_prob = F.pad(fft_prob, (0, padding_size), "constant", 0)

            # Use the probabilities to weight
            #  the embeddings
            weighted_embeddings = transformed_seq_embeddings[i] * fft_prob.view(-1, 1)

            # Aggregate the embeddings by summing them
            aggregated_embedding = weighted_embeddings.sum(dim=0)

            # Add the aggregated embedding to the list
            aggregated_embeddings.append(aggregated_embedding)

        # Convert the list to a tensor with shape [16, 128]
        aggregated_embeddings = torch.stack(aggregated_embeddings)

        return aggregated_embeddings #[16, 128]

class wave_Layer(nn.Module):
    def __init__(self, k=5):
        super(wave_Layer, self).__init__()
        self.k = k  # 采样次数，默认为5次

    def forward(self, seqs, seq_embeddings):
        # [16, 49], [16, 49, 128]
        aggregated_embeddings = []

        # For each sample
        for i in range(seqs.shape[0]):
            seqMask = [seqs[i] != 0]
            real_poi = seqs[i][seqMask]
            binary_seq = (real_poi == real_poi[-1]).float()  # [46,]

            # 存储所有采样的权重
            all_fft_probs = []

            for j in range(self.k):
                # 2. 进行小波变换
                coeffs = pywt.wavedec(binary_seq, 'db1')  # db1 is Haar wavelet

                # 3. 逆小波变换
                fft_prob = pywt.waverec(coeffs, 'db1')
                fft_prob = fft_prob[:len(binary_seq)]

                # 将 fft_prob 转换为 PyTorch Tensor
                fft_prob = torch.tensor(fft_prob, dtype=torch.float32)

                fft_prob = torch.abs(fft_prob)
                fft_prob = fft_prob / fft_prob.sum()
                fft_prob[-(j + 1):] = 0  # 将末尾 j+1 个位置设置为 0

                # Shift the probabilities to the left by j+1 positions
                fft_prob = torch.roll(fft_prob, shifts=j + 1)  # [46,] 个概率

                # 将当前采样的权重添加到列表中
                all_fft_probs.append(fft_prob)

                # 更新 binary_seq 以倒数 j+2 个 POI 作为参考
                if j + 1 < len(real_poi):
                    binary_seq = (real_poi == real_poi[-(j + 2)]).float()

            # 将所有采样的权重求和平均
            avg_fft_prob = torch.stack(all_fft_probs).mean(dim=0)

            padding_size = seqs.shape[1] - avg_fft_prob.shape[0]  # Calculate the padding size
            avg_fft_prob = F.pad(avg_fft_prob, (0, padding_size), "constant", 0)

            # Use the averaged probabilities to weight the embeddings
            weighted_embeddings = seq_embeddings[i] * avg_fft_prob.view(-1, 1)  # [49, 128]

            # Aggregate the embeddings by summing them
            aggregated_embedding = weighted_embeddings.sum(dim=0)  # [128]

            # Add the aggregated embedding to the list
            aggregated_embeddings.append(aggregated_embedding)

        # Convert the list to a tensor with shape [16, 128]
        aggregated_embeddings = torch.stack(aggregated_embeddings)

        return aggregated_embeddings  # [16, 128]

class wave_Layer_ed(nn.Module):
    def __init__(self):
        super(wave_Layer, self).__init__()
        # Initialize your layers here

    def forward(self, seqs,     seq_embeddings):
        #           [16, 49],   [16, 49, 128]    
        aggregated_embeddings = []

        # For each sample
        for i in range(seqs.shape[0]):
            
            seqMask=[seqs[i] != 0]
            real_poi = seqs[i][seqMask]
            # Create a binary sequence indicating whether the last POI was visited at each time point
            binary_seq = (real_poi == real_poi[-1]).float() #[46,]


            # 2. 进行小波变换
            coeffs = pywt.wavedec(binary_seq, 'db1')  # db1 is Haar wavelet

            # 3. 逆小波变换
            fft_prob = pywt.waverec(coeffs, 'db1')

            fft_prob=fft_prob[:len(binary_seq)]

            # 将 fft_prob 转换为 PyTorch Tensor
            fft_prob = torch.tensor(fft_prob, dtype=torch.float32)


            
            fft_prob = torch.abs(fft_prob)
            fft_prob = fft_prob / fft_prob.sum()
            fft_prob[-1] = 0
       
            # Shift the probabilities to the left by one position
            fft_prob = torch.roll(fft_prob, shifts=1)#[46,]个概率

            padding_size = seqs.shape[1] - fft_prob.shape[0]  # Calculate the padding size
            fft_prob = F.pad(fft_prob, (0, padding_size), "constant", 0)

            # Use the probabilities to weight
            #  the embeddings
            weighted_embeddings = seq_embeddings[i] * fft_prob.view(-1, 1) #[49, 128]
            # weighted_embeddings = transformed_seq_embeddings[i]

            # Aggregate the embeddings by summing them
            aggregated_embedding = weighted_embeddings.sum(dim=0) #[128]

            # Add the aggregated embedding to the list
            aggregated_embeddings.append(aggregated_embedding)

        # Convert the list to a tensor with shape [16, 128]
        aggregated_embeddings = torch.stack(aggregated_embeddings)

        return aggregated_embeddings #[16, 128]



class Attentional_Aggregation(nn.Module):
    def __init__(self, emb_dim):
        super(Attentional_Aggregation, self).__init__()
        self.query_layer = nn.Linear(emb_dim, emb_dim)
        self.key_layer = nn.Linear(emb_dim, emb_dim)

    def forward(self, embeddings, sampleIdOfDay_set):
        unique_ids = torch.unique(sampleIdOfDay_set)
        group_embeddings = []

        for uid in unique_ids:
            # Get the embeddings of the current group
            group_emb = embeddings[sampleIdOfDay_set == uid] #[6, 128] uid==0时 有6天

            # Compute the query (the last embedding of the group)
            query = self.query_layer(group_emb[-1]).unsqueeze(0)  #[1, 128] 最后一天,短期,重要,作为Query

            # Compute the keys
            keys = self.key_layer(group_emb) #[6, 128] 

            # Compute the attention scores
            scores = F.softmax(query @ keys.transpose(-2, -1), dim=-1) #[1, 6]

            # Compute the final embedding of the group
            group_emb_final = scores @ keys #[1, 128]

            group_embeddings.append(group_emb_final)

        group_embeddings = torch.cat(group_embeddings, dim=0) #[32, 128]

        return group_embeddings #[32, 128]






class Prediction_Layer(nn.Module):
    def __init__(self, emb_dim, poi_num, dropout):
        super(Prediction_Layer, self).__init__()
        self.linear = nn.Linear(emb_dim, poi_num)
        self.out_linear = nn.Sequential(nn.Linear(emb_dim, emb_dim*2),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(emb_dim*2, poi_num))

    def forward(self, inputs):
        prediction=self.linear(inputs)
        # out = user_embeddings.matmul(embeddings.T[:, 1:])

        return prediction






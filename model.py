import torch
from torch import nn
from Layer import *
from Tools import *
# device = settings.gpuId if torch.cuda.is_available() else 'cpu'
from testmamba import ModelArgs, Mamba

class PSPLC(nn.Module):
    def __init__(self,args,mamba_args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.user_num = args.user_num
        self.poi_num = args.poi_num
        self.device = args.device
        self.GCN_layer_num=args.GCN_layer_num

        self.poi_embedding = nn.Embedding(args.poi_num+1, args.poi_dim, padding_idx=0) #[1431,64]
        
        self.Personal_Geo_GCN = Personal_Geo_GCN(args.poi_dim,args.GCN_layer_num,args.device)

        self.Embedding_Layer = Embedding_Layer(args)

        self.Distance_Time_Span_Embedding_Layer = Distance_Time_Span_Embedding_Layer(args)

        self.SelfAttention_Layer = SelfAttention_Layer(args.hidden_dim,args.dropout_rate)

        self.Span_LSTM_Layer=Span_LSTM_Layer(args.hidden_dim,args.hidden_dim)

        self.LSTM_Layer = LSTM_Layer(args.hidden_dim, args.hidden_dim,args.dropout_rate)


        self.SeqMamba_Module=SeqMamba_Module(mamba_args)

        self.DFT_layer=DFT_Layer()

        self.wave_layer=wave_Layer()

        self.Attentional_Aggregation = Attentional_Aggregation(args.hidden_dim)

        self.Prediction_Layer=Prediction_Layer(args.hidden_dim, args.poi_num, args.dropout_rate)

    def forward_with_negetive_sample(self, args,seqs, distance_matrix, n_seqs,
                                         distance_span_set,    time_span_set,   seqInDay,  hour_set,   user_set,    isweekend_set,   mask ,   sampleIdOfDay_set,
                                        n_distance_span_set, n_time_span_set, n_seqInDay, n_hour_set, n_user_set, n_isweekend_set, n_mask , n_sampleIdOfDay_set):
        
        ori_poi_embeddings=self.poi_embedding(seqInDay)
        seq_embeddings=self.Embedding_Layer(ori_poi_embeddings, hour_set,isweekend_set,user_set)# [157, 15, 128]
       


        #Short-Term Preference Encoder  =============================================================================
        seq_short_embeddings=seq_embeddings
        
        #Personalized Spatial Span GCN
        if args.is_GCN:
            new_poi_embeddings= self.Personal_Geo_GCN(self.poi_embedding, distance_matrix, seqInDay, self.GCN_layer_num,sampleIdOfDay_set)# [80, 15, 64] 使用GCN得到的poiemb
            hour_isweekend_user_embeddings = seq_embeddings[:, :, args.poi_dim:]  # [80, 15, 64]
            seq_short_embeddings = torch.cat((new_poi_embeddings, hour_isweekend_user_embeddings), dim=2)  # [80, 15, 128]

        #ASL Block
        seq_short_embeddings=self.SelfAttention_Layer(seq_short_embeddings,mask)# [80, 15, 128]
        if args.is_SLSTM:
            distance_span_embedding, time_span_embedding=self.Distance_Time_Span_Embedding_Layer(distance_span_set, time_span_set)
            seq_short_embeddings,day_embeddings=self.Span_LSTM_Layer(seq_short_embeddings,distance_span_embedding, time_span_embedding,mask)
        else:
            seq_short_embeddings,day_embeddings=self.LSTM_Layer(seq_short_embeddings,mask) # [80, 15, 128], [80, 128]

        short_term_embeddings=self.Attentional_Aggregation(day_embeddings,sampleIdOfDay_set) # [16, 128]
        # end Short-Term Preference Encoder ===============================================================================================




        # Long-Term Preference Encoder  ===============================================================================
        
        #正样本
        seq_long_embeddings=seq_embeddings #[80, 15, 128]
        t_seq_long_embeddings=transforme_seq(seqs,seq_long_embeddings,mask , sampleIdOfDay_set) #[16, 49, 128])
        t_seq_long_embeddings=self.SeqMamba_Module(t_seq_long_embeddings) #[16, 49, 128])
        long_term_embeddings=self.DFT_layer(seqs,t_seq_long_embeddings) # [16, 128]
        # long_term_embeddings=self.wave_layer(seqs,t_seq_long_embeddings)
        
        
        #负样本
        n_seq_poi_embeddings=self.poi_embedding(n_seqInDay) #[300, 15, 64]
        n_seq_embeddings=self.Embedding_Layer(n_seq_poi_embeddings, n_hour_set,n_isweekend_set,n_user_set)# [300, 15, 128]
        t_n_seq_long_embeddings=transforme_seq(n_seqs,n_seq_embeddings, n_mask , n_sampleIdOfDay_set) #[64, 64, 128]
        t_n_seq_long_embeddings=self.SeqMamba_Module(t_n_seq_long_embeddings) #[64, 64, 128]
        neg_long_term_embeddings=self.DFT_layer(n_seqs,t_n_seq_long_embeddings) #[64, 128]
        # neg_long_term_embeddings=self.wave_layer(n_seqs,t_n_seq_long_embeddings)
        
        # 将 neg_long_term_embeddings 重新形状为 [16, 4, 128]
        neg_long_term_embeddings = neg_long_term_embeddings.view(-1, args.neg_sample_count, neg_long_term_embeddings.size(1))
        # 对每个样本的负采样求和平均，得到 [16, 128]
        neg_long_term_embeddings = neg_long_term_embeddings.mean(dim=1) #[16, 128]

        # end Long-Term Preference Encoder ===============================================================================================



        prediction=self.Prediction_Layer(long_term_embeddings+short_term_embeddings)# [32, 1430]
        con_loss=contrast_learning_loss(short_term_embeddings, long_term_embeddings, neg_long_term_embeddings,self.device)
        return prediction,con_loss, short_term_embeddings,long_term_embeddings




    def forward(self,args, seqs,distance_matrix, distance_span_set, time_span_set, seqInDay,  hour_set,    user_set, isweekend_set,    mask ,   sampleIdOfDay_set):
                #       [157, 15],  [157, 15],                                    [157, 15],   [157,15],    [157,15],       [157,]   


        ori_poi_embeddings=self.poi_embedding(seqInDay)
        seq_embeddings=self.Embedding_Layer(ori_poi_embeddings, hour_set,isweekend_set,user_set)# [157, 15, 128]
       

        #Short-Term Preference Encoder  =============================================================================
        seq_short_embeddings=seq_embeddings
        
        #Personalized Spatial Span GCN
        if args.is_GCN:
            new_poi_embeddings= self.Personal_Geo_GCN(self.poi_embedding, distance_matrix, seqInDay, self.GCN_layer_num,sampleIdOfDay_set)# [157, 15, 64] 使用GCN得到的poiemb
            hour_isweekend_user_embeddings = seq_embeddings[:, :, args.poi_dim:]  # [157, 15, 64]
            seq_short_embeddings = torch.cat((new_poi_embeddings, hour_isweekend_user_embeddings), dim=2)  # [157, 15, 128]

        #ASL Block
        seq_short_embeddings=self.SelfAttention_Layer(seq_short_embeddings,mask)# [157, 15, 128]
        if args.is_SLSTM:
            distance_span_embedding, time_span_embedding=self.Distance_Time_Span_Embedding_Layer(distance_span_set, time_span_set)
            seq_short_embeddings,day_embeddings=self.Span_LSTM_Layer(seq_short_embeddings,distance_span_embedding, time_span_embedding,mask)
        else:
            seq_short_embeddings,day_embeddings=self.LSTM_Layer(seq_short_embeddings,mask) # [157, 15, 128], [157, 128]

        short_term_embeddings=self.Attentional_Aggregation(day_embeddings,sampleIdOfDay_set) # [32, 128]
        # end Short-Term Preference Encoder ===============================================================================================



        # Long-Term Preference Encoder  ===============================================================================
        seq_long_embeddings=seq_embeddings #[80, 15, 128]
        t_seq_long_embeddings=transforme_seq(seqs,seq_long_embeddings,mask , sampleIdOfDay_set) #[16, 49, 128])
        t_seq_long_embeddings=self.SeqMamba_Module(t_seq_long_embeddings) #[16, 49, 128])
        long_term_embeddings=self.DFT_layer(seqs,t_seq_long_embeddings) # [16, 128]
        # long_term_embeddings=self.wave_layer(seqs,t_seq_long_embeddings) # [16, 128]
        
        # end Long-Term Preference Encoder ===============================================================================================
       
       
        prediction=self.Prediction_Layer(long_term_embeddings+short_term_embeddings)# [32, 1430]
        return prediction,short_term_embeddings,long_term_embeddings

    
    
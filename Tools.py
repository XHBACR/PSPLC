

import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import TrainDataset
def padSeq(seq_set):
    xMax = max(len(sublist) for sublist in seq_set)
    # print("max len of seq : ",xMax)
    new_list = [sublist[:] for sublist in seq_set]
    mask_matrix = torch.tensor([[1 if i < len(sublist) else 0 for i in range(xMax)] for sublist in new_list])
    for sublist in new_list:
        sublist.extend([-1] * (xMax - len(sublist)))
    new_list=torch.tensor(new_list)
    return new_list,mask_matrix

def calculate_spans(distance_matrix, min_distance, seqofday, hour_set):
    # Initialize the distance and time spans with the first element as 0
    distance_span = [0]
    time_span = [0]

    # Calculate the distance and time spans
    for i in range(1, len(seqofday)):
        # Calculate the distance span
        distance = (distance_matrix[seqofday[i-1]][seqofday[i]]).to('cpu')
        distance_span.append(int(min(255, np.floor(max(distance,min_distance) / min_distance))))

        # Calculate the time span
        time_span.append(int(max(0,np.floor(hour_set[i] - hour_set[i-1]))))

    return distance_span, time_span


def fixData(batch,distance_matrix,min_distance,device):
    seqInDay_set=[]
    cat_set=[]
    user_set=[]
    hour_set=[]
    isweekend_set=[]

    sampleIdOfDay_set=[]

    target=[]
    distance_span_set=[]
    time_span_set=[]


    seqs=[]
    for sample_id,sample in enumerate(batch): 
        
        seqsOfSample=[]
        for id,day in enumerate(sample): 
            if id==len(sample)-1: 
                target.append(day[0][-1])
                seqInDay_set.append(day[0][:-1])
                seqsOfSample.append(day[0][:-1])
                cat_set.append(day[1][:-1])
                user_set.append(day[2][:-1])
                hour_set.append(day[3][:-1])
                distance_span, time_span=calculate_spans(distance_matrix, min_distance, day[0][:-1], day[3][:-1])

                distance_span_set.append(distance_span)
                time_span_set.append(time_span)

                isweekend_set.append(day[4][:-1])
                sampleIdOfDay_set.append(sample_id)

                # seqs.append(np.concatenate(seqsOfSample))
                seqs.append([item for sublist in seqsOfSample for item in sublist])
                continue
            seqInDay_set.append(day[0])
            seqsOfSample.append(day[0])
            cat_set.append(day[1])
            user_set.append(day[2])
            hour_set.append(day[3])
            distance_span, time_span=calculate_spans(distance_matrix, min_distance, day[0], day[3])

            distance_span_set.append(distance_span)
            time_span_set.append(time_span)

            isweekend_set.append(day[4])
            sampleIdOfDay_set.append(sample_id) 

            
    sampleIdOfDay_set=torch.tensor(sampleIdOfDay_set)

    seqInDay,mask= padSeq(seqInDay_set)
    seqs,_= padSeq(seqs)

    cat_set,_= padSeq(cat_set)
    hour_set,_= padSeq(hour_set)
    user_set,_=padSeq(user_set)
    isweekend_set,_=padSeq(isweekend_set)
    distance_span_set,_=padSeq(distance_span_set)
    time_span_set,_=padSeq(time_span_set)

    target=torch.tensor(target)

    return (seqInDay+1).to(device), (seqs+1).to(device), (distance_span_set+1).to(device), (time_span_set+1).to(device), target.to(device),  (cat_set+1).to(device)  ,(hour_set+1).to(device)  ,(user_set+1).to(device)  ,(isweekend_set+1).to(device)  ,mask.to(device)  ,sampleIdOfDay_set.to(device) 
     
def fix_negativeData(batch, distance_matrix, min_distance, device):
    seqInDay_set = []
    cat_set = []
    user_set = []
    hour_set = []
    isweekend_set = []
    distance_span_set = []
    time_span_set = []
    sampleIdOfDay_set = []
    seqs = []


    for sample_id, sample in enumerate(batch): 
        
        seqsOfSample = []
        for id, day in enumerate(sample): 
            seqInDay_set.append(day[0])
            seqsOfSample.append(day[0])
            cat_set.append(day[1])
            user_set.append(day[2])
            hour_set.append(day[3])

            distance_span, time_span = calculate_spans(distance_matrix, min_distance, day[0], day[3])
            distance_span_set.append(distance_span)
            time_span_set.append(time_span)

            isweekend_set.append(day[4])
            sampleIdOfDay_set.append(sample_id)  

        seqs.append([item for sublist in seqsOfSample for item in sublist])

    sampleIdOfDay_set = torch.tensor(sampleIdOfDay_set)

    seqInDay, mask = padSeq(seqInDay_set)
    seqs, _ = padSeq(seqs)
    cat_set, _ = padSeq(cat_set)
    hour_set, _ = padSeq(hour_set)
    user_set, _ = padSeq(user_set)
    isweekend_set, _ = padSeq(isweekend_set)
    distance_span_set, _ = padSeq(distance_span_set)
    time_span_set, _ = padSeq(time_span_set)

    return (seqInDay + 1).to(device), (seqs + 1).to(device), (distance_span_set + 1).to(device), (time_span_set + 1).to(device), (cat_set + 1).to(device), (hour_set + 1).to(device), (user_set + 1).to(device), (isweekend_set + 1).to(device), mask.to(device), sampleIdOfDay_set.to(device) 


def transforme_seq(   seqs,     seq_embeddings,   mask , sampleIdOfDay_set):


    max_length = seqs.shape[1]

    # Initialize an empty list to store the transformed sequence embeddings
    transformed_seq_embeddings = []

    # For each unique sample ID
    unique_sample_ids = torch.unique(sampleIdOfDay_set)
    for sample_id in unique_sample_ids:
        # Get the sequence embeddings and mask for this sample
        sample_seq_embeddings = seq_embeddings[sampleIdOfDay_set == sample_id] 
        sample_mask = mask[sampleIdOfDay_set == sample_id] 

        # Remove the padding POIs and concatenate the remaining POIs
        sample_seq_embeddings = sample_seq_embeddings[sample_mask.bool()].view(-1, 128) 

        # If the number of POIs is less than max_length, pad with zero vectors
        if sample_seq_embeddings.shape[0] < max_length:
            padding = torch.zeros((max_length - sample_seq_embeddings.shape[0], 128), device=sample_seq_embeddings.device)
            sample_seq_embeddings = torch.cat([sample_seq_embeddings, padding], dim=0)

        # Add the transformed sequence embeddings to the list
        transformed_seq_embeddings.append(sample_seq_embeddings)

    transformed_seq_embeddings = torch.stack(transformed_seq_embeddings) 
    return transformed_seq_embeddings


def fix_negativeData_ed(batch,distance_matrix, min_distance,device):
    seqInDay_set=[]
    cat_set=[]
    user_set=[]
    hour_set=[]
    isweekend_set=[]
    distance_span_set=[]
    time_span_set=[]

    sampleIdOfDay_set=[]

    for sample_id,sample in enumerate(batch): 
        for id,day in enumerate(sample): 
            
            seqInDay_set.append(day[0])
            cat_set.append(day[1])
            user_set.append(day[2])
            hour_set.append(day[3])

            distance_span, time_span=calculate_spans(distance_matrix, min_distance, day[0], day[3])
            distance_span_set.append(distance_span)
            time_span_set.append(time_span)
            
            isweekend_set.append(day[4])
            sampleIdOfDay_set.append(sample_id) 

            
    sampleIdOfDay_set=torch.tensor(sampleIdOfDay_set)

    seqInDay,mask= padSeq(seqInDay_set)
    cat_set,_= padSeq(cat_set)
    hour_set,_= padSeq(hour_set)

    distance_span_set,_=padSeq(distance_span_set)
    time_span_set,_=padSeq(time_span_set)
    user_set,_=padSeq(user_set)

    isweekend_set,_=padSeq(isweekend_set)

    return (seqInDay+1).to(device), (distance_span_set+1).to(device), (time_span_set+1).to(device), (cat_set+1).to(device),  (hour_set+1).to(device)  ,(user_set+1).to(device)  ,(isweekend_set+1).to(device)  ,mask.to(device),sampleIdOfDay_set.to(device)  #用-1padding的,全部+1,即用0padding

def generate_negative_sample_list(train_set, batch, neg_sample_count):
    neg_day_samples = []
    for sample in batch:
        user_id = sample[0][2][0]
        # Random sample k negative samples from other users' trajectories
        neg_day_sample = random.sample([seq for seq in train_set if seq[0][2][0] != user_id], neg_sample_count)
        neg_day_samples.extend(neg_day_sample)  

    return neg_day_samples


def neg_sample_loss_function(last_embedding, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def single_infoNCE_loss_simple(last_embedding, neg_embedding):

            neg_score = score(last_embedding, neg_embedding)

            # one = torch.cuda.FloatTensor([1], device=device)
            # one = torch.FloatTensor([1], device=device)
            one = torch.tensor([1.])
            con_loss = torch.sum(- torch.log(1e-8 + (one - torch.sigmoid(neg_score))))
            return con_loss

        neg_loss = single_infoNCE_loss_simple(last_embedding, neg_embedding)
        return neg_loss

def contrast_learning_loss(short_emb, long_emb,  neg_emb,device):
    def score(x1, x2):
        return torch.mean(torch.mul(x1, x2))
    def cos(x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)

    pos=cos(short_emb, long_emb)
    pos[pos<0.4]=0
    pos=pos.mean()
    neg = cos(long_emb, neg_emb).mean()

    one = torch.tensor([1.],device=device)
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
    return con_loss


def matrix_mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj

def trainSet_to_seqMatrix(train_set,poi_num):
    seqInDay_set=[]

    for sample_id,sample in enumerate(train_set): 
        for id,day in enumerate(sample): 
            seqInDay_set.append(day[0])

    # Initialize the matrix with zeros
    matrix = torch.zeros(poi_num, poi_num)

    # Iterate over each day in seqInDay_set
    for day in seqInDay_set:
        # Iterate over each checkin in the day
        for i in range(len(day) - 1):
            # If there is a next checkin
            if i + 1 < len(day):
                # Increment the corresponding matrix value
                matrix[day[i], day[i+1]] += 1

    row_maxs, _ = torch.max(matrix, dim=1, keepdim=True)
    matrix = torch.where(row_maxs != 0, matrix / row_maxs, matrix)
    identity = torch.eye(matrix.size(0))
    matrix = matrix + identity  

    return matrix 

def test_model(args, distance_matrix,min_distance,test_set, model,logger, ks ):

    def calc_recall(labels, preds, k): 
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1  
        ndcg = 1 / torch.log2(exist_pos + 1) 
        return torch.sum(ndcg) / labels.shape[0] 

    def calc_map(labels, preds, k): 
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1 
        map = 1 / exist_pos 
        return torch.sum(map) / labels.shape[0] 
    
    test_dataset=TrainDataset(test_set)
    batch_num = len(test_dataset) // args.batch_size
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,collate_fn=lambda x: x)

    targets=[]
    rankings=[]
    for batch_id,batch in tqdm(enumerate(test_dataloader),total=batch_num,ncols=100,leave=False):
        seqInDay,  seqs, distance_span_set, time_span_set,  target,   cat_set,   hour_set,   user_set,  isweekend_set,    mask  ,sampleIdOfDay_set =fixData(batch,distance_matrix,min_distance,args.device)
        prediction,_,_=model(args,seqs,distance_matrix,distance_span_set, time_span_set, seqInDay, hour_set,   user_set,  isweekend_set,  mask, sampleIdOfDay_set) #[38, 1431]
        target=target.unsqueeze(1) 
        targets.append(target)

        ranking = torch.sort(prediction, descending=True)[1]
        rankings.append(ranking)

    target = torch.cat(targets, dim=0)
    ranking = torch.cat(rankings, dim=0) 
    recalls, NDCGs, MAPs = {}, {}, {}
    logger.info("Test results: ")
    for k in ks: 
        recalls[k] = calc_recall(target, ranking, k)
        NDCGs[k] = calc_ndcg(target, ranking, k)
        # MAPs[k] = calc_map(labels, preds, k)
        # print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")
        
        logger.info(f"Recall @{k:02} : {recalls[k]:10.5f},\tNDCG@{k:02} : {NDCGs[k]:10.5f}")
        
        # print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]}",file=f)
    logger.info(" \n")

    return recalls, NDCGs






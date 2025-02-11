import argparse
import datetime
import logging
import numpy as np
from tqdm import tqdm
import torch
import pickle
import time
import os
import pandas as pd
from torch.utils.data import DataLoader
from Dataset import *
from Tools import *
from model import *
import pandas as pd
from testmamba import ModelArgs, Mamba

if __name__ == '__main__':
    
    current_path='./'
    parser = argparse.ArgumentParser()

    parser.add_argument('--city', default='PHO')
    parser.add_argument('--is_GCN', default=0, type=bool)
    parser.add_argument('--is_Con', default=0, type=bool)
    parser.add_argument('--is_SLSTM', default=1, type=bool)
    parser.add_argument('--GCN_layer_num', default=1, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--neg_sample_count', default=4, type=int)
    parser.add_argument('--poi_num', default=1430, type=int)
    parser.add_argument('--user_num', default=168, type=int)
    parser.add_argument('--hour_num', default=24, type=int)
    parser.add_argument('--isweekend_num', default=2, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--poi_dim', default=64, type=int)
    parser.add_argument('--user_dim', default=32, type=int)
    parser.add_argument('--hour_dim', default=16, type=int)
    parser.add_argument('--isweekend_dim', default=16, type=int)
    args = parser.parse_args()

    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    


    # Convert to string and use as file name
    filename = now_str + '.txt'
    directory = current_path+'logs/temp/'
    full_path = directory + filename


    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(full_path)
    file_handler.setLevel(logging.INFO)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Current time: {now_str}")


    # processed_data_directory = current_path+'processed_data/original'
    processed_data_directory = current_path+'my_data/processed'

    # Read training data
    file = open(f"{processed_data_directory}/{args.city}_train", 'rb')
    train_set = pickle.load(file)
    

    file = open(f"{processed_data_directory}/{args.city}_valid", 'rb')
    valid_set = pickle.load(file)

    # Read meta data
    file = open(f"{processed_data_directory}/{args.city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI":len(meta["POI"]),
                  "cat":len(meta["cat"]),
                  "user":len(meta["user"]),
                  "hour":len(meta["hour"]),
                  "day":len(meta["day"])}
    args.poi_num = vocab_size["POI"]
    args.user_num = vocab_size["user"]

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device=device

    distance_matrix= pd.DataFrame()
    # Read the CSV file
    distance_matrix = pd.read_csv(f"{current_path}my_data/raw/{args.city}_poi_distance.csv", header=None,dtype=np.float32)
    # Convert the DataFrame to a PyTorch tensor
    distance_matrix = torch.tensor(distance_matrix.values)
    distance_matrix=distance_matrix.to(device)
    distance_matrix=distance_matrix[1:]
    min_distance = 0.1

    for arg, value in vars(args).items():
      logger.info(f"{arg:20} : {value}")

    logger.info(f'Current Device {args.device} \n')
    train_dataset=TrainDataset(train_set)
    batch_num = len(train_dataset) // args.batch_size
    trian_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,collate_fn=lambda x: x)


    mamba_args = ModelArgs()
    model=PSPLC(args,mamba_args)
    prediction_loss_function = torch.nn.CrossEntropyLoss()
    graph_loss_function = torch.nn.KLDivLoss(reduction="batchmean")
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    model=model.to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass

    

    
    
    for epoch in range(0, args.epoch_num):
        model.train()
        begin_time = time.time()
        total_loss = 0.
        con_loss=0.
        for batch_id,batch in tqdm(enumerate(trian_dataloader),total=batch_num,ncols=100,leave=False):
            seqInDay,   seqs,            distance_span_set, time_span_set, target,   cat_set,   hour_set,   user_set,  isweekend_set,    mask  ,sampleIdOfDay_set =fixData(batch,distance_matrix,min_distance,args.device) # "所有"编号都+1了,以便用0 padding
            
            if args.is_Con: #使用不使用对比学习
                negative_sample=generate_negative_sample_list(train_set, batch, args.neg_sample_count)
                n_seqInDay, n_seqs, n_distance_span_set, n_time_span_set,n_cat_set, n_hour_set, n_user_set, n_isweekend_set, n_mask , n_sampleIdOfDay_set=fix_negativeData(negative_sample,distance_matrix,min_distance,args.device)
                # 使用对比学习:
                prediction,con_loss,_,_=model.forward_with_negetive_sample(args, seqs, distance_matrix, n_seqs, distance_span_set, time_span_set,   seqInDay, hour_set,   user_set,  isweekend_set,  mask, sampleIdOfDay_set ,    n_distance_span_set, n_time_span_set, n_seqInDay, n_hour_set, n_user_set, n_isweekend_set, n_mask , n_sampleIdOfDay_set)
                
            else:
                # 不用对比学习
                prediction ,_ ,_=model(args,seqs,distance_matrix,distance_span_set, time_span_set, seqInDay, hour_set,   user_set,  isweekend_set,  mask, sampleIdOfDay_set) #[32, 1430]
                
            prediction_loss=prediction_loss_function(prediction,target)
            loss=prediction_loss+ con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / batch_num

        logger.info(f"Epoch {epoch} finished, avg_loss:{avg_loss:.5f}, total loss: {total_loss:.5f}, time: {time.time()-begin_time:.2f}")
        # print(f"\nEpoch {epoch} finished,avg_loss:{avg_loss}, total loss: {total_loss}, time: {time.time()-begin_time}")
        model.eval() 
        test_model(args, distance_matrix,min_distance,valid_set, model,logger, ks=[1, 5, 10])

        # model_save_path = f"./savedModel/{args.city}_model.pth"
        # torch.save(model.state_dict(), model_save_path)
        # logger.info(f"Model saved to {model_save_path}")

    
    
    

    
    
    print("Finished!")











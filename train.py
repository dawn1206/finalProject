import os

import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import random
import metrics
from torch.utils.data import Dataset, DataLoader
from convCal import TransAm,Trans,resNet,vit,swinT,effNet,Alex
from astmodel import ASTModel
from dataLoader import get_data
import torch.onnx

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
calculate_loss_over_all_values = False
input_window = 100
output_window = 5
batch_size = 64# batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.cuda.set_device('cuda:0,1,2,3')
print(device)
global train_data, val_data,model,scheduler,epochs,criterion,optimizer

def modelInitialization(modelName):
    if modelName == "Trans":
        model = Trans().to(device)
    elif modelName == "Alex":
        model = Alex().to(device)
    elif modelName == "vit":
        model = vit().to(device)
    elif modelName == "swinT":
        model = swinT().to(device)
    elif modelName == "resNet":
        model = resNet().to(device)
    elif modelName == "effNet":
        model = effNet().to(device)
    elif modelName == "ssast":
        model = ASTModel(label_dim=4,
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=399, input_tdim=213, model_size='base',
                 pretrain_stage=False).to(device)
    else:
         raise AssertionError("model not exists")
    return model

def main(modelName):
    train_data, val_data = get_data(load=False,batch_size=batch_size)
    model = modelInitialization(modelName)
    criterion = nn.CrossEntropyLoss()
    lr = 5e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
    best_acc = 0
    best_val_loss = float("inf")
    epochs = 200  # The number of epochs
    best_model = None
    import pandas as pd
    df = pd.DataFrame(columns=["acc","loss","recall","precision","auc","f1"],index=range(1,201))
    for epoch in range(1, epochs + 1):
        print(epoch)
        val_loss,val_acc = metrics.test_loop(val_data,model,criterion,epoch,df,"metrics"+modelName)
        if val_acc > best_acc:
            torch.save(model.state_dict(), "bestmodel.pth")
        model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()

        for batch, (input_data, targets) in enumerate(train_data):
            # torch.onnx.export(model,input_data ,"my_model.pth")
            # print('input_data%d' % batch, input_data)
            # print('target%d' % batch, targets)
            optimizer.zero_grad()
            output = model(input_data)

            loss = criterion(output, targets.long().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 4
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.6f} | {:5.2f} ms | '
                    'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            epoch_start_time = time.time()
        

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                    time.time() - epoch_start_time),val_loss,math.exp(val_loss)))
        print('-' * 89)
        
        scheduler.step()
        if epoch % 10 ==0:
            df.to_csv(modelName+"2s.csv")
        
        
    
if __name__ == "__main__":
    # trainVIT()
    #trainEFF()
    main("effNet")

# if __name__ == "__main__":
#     # get_data()
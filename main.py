import argparse
import math
import time
from models import MSPNet
import torch.nn as nn
from matplotlib import pyplot as plt
from Optim import Optim
from utils import *

def evaluate(data, X,Y_his, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y_his ,Y in data.get_batches(X,Y_his,Y, batch_size, False):
        output = model(X,Y_his)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        total_loss += evaluateL2(output , Y).item()
        total_loss_l1 += evaluateL1(output , Y).item()
        n_samples += (output.size(0) * args.horizon)
    MAE=total_loss_l1/n_samples
    MSE=total_loss/n_samples
    RMSE=math.sqrt(total_loss/n_samples)
    return  MAE,MSE,RMSE

def train(data, X,Y_his, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y_his,Y in data.get_batches(X, Y_his,Y, batch_size, True):
        model.zero_grad()
        output = model(X,Y_his)
        loss = criterion(output, Y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        n_samples += (output.size(0)*args.horizon)
    return total_loss / n_samples


#electricity:  17714,2609,5261
#traffic1:  11513,1755,3509
#traffic2:  9494,1459,2919
#CAISO:  17703,2632,5264

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--dataset', type=str, default='traffic1',   help=' data name')
parser.add_argument('--data', type=str, default=r'./datasets/traffic1.txt',
                    help='location of the data file')  # required=True,
parser.add_argument('--model', type=str, default='MSPNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=50,
                    help='number of CNN hidden units')
parser.add_argument('--CNN_kernel', type=int, default=5,
                    help='the kernel size of the CNN layers')
parser.add_argument('--hidRNN', type=int, default=50,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24,
                    help='window size')
parser.add_argument('--highway_window', type=int, default=23, #In order to use one-step prediction, B must be at least 1 smaller than A
                    help='The window size of the AR component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=str, default="cuda:0")
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/MSPNet.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--horizon', type=int, default=24 ,help='pred_len')
parser.add_argument('--periods',type=list,default=[24],help='length of each period(season).'
                                                            'When the length is a multiple, generally enter the smallest one.')
parser.add_argument('--ptl',type=list,default=[30] ,help='The number of inputs corresponding to each period')    #For example, periods=[24], ptl=[30], the actual input matrix is 30*24
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--output_fun', type=str, default='None')
parser.add_argument('--train_len', type=int, default=11513)
parser.add_argument('--valid_len', type=int, default=1755)
parser.add_argument('--train_len_proportion', type=float, default=0.7)
parser.add_argument('--valid_len_proportion', type=float, default=0.1)
parser.add_argument('--divide_data', type=int, default=1,
                    help='If divide_data=1, divide the data set according to the specific value, if divide_data=2, divide the data set according to the proportion')
parser.add_argument('--pred_method', type=int, default=0,
                    help='If the value is 1, the direct method is used, if the value is not 1, the generative method is used')
parser.add_argument('--inverse', type=int, default=1,help='inverse normalized.The value of 1 means Yes, otherwise it is No')
args = parser.parse_args()

if args.cuda:
    torch.cuda.set_device(args.gpu)

if args.divide_data==1:
    Data=Data_Utility(args.data,args.train_len,args.valid_len,args.divide_data,args.horizon,args.window,args.periods,args.ptl,args.cuda,args.dataset,args.pred_method)
else:
    Data = Data_Utility(args.data, args.train_len_proportion, args.valid_len_proportion, args.divide_data, args.horizon, args.window,args.periods, args.ptl, args.cuda,args.dataset,args.pred_method)
print(Data.train[2].shape,Data.valid[2].shape,Data.test[2].shape)


# print(Data.train[2].shape[0],Data.valid[2].shape[0],Data.test[2].shape[0])
# print(Data.raw_data.shape)
# print(Data.train[2].shape)
# print(Data.train[2])


model = eval(args.model).Model(args, Data)
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

best_val = 10000000
optim = Optim(
    model.parameters(), 'adam', args.lr, args.clip, start_decay_at = 10, lr_decay = 0.9
)

train_loss_set=[]

try:
    print('Start training....')
    for epoch in range(1, args.epochs + 1):                      #start training
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1],Data.train[2], model, criterion, optim, args.batch_size)
        MAE,MSE,RMSE = evaluate(Data, Data.valid[0], Data.valid[1],Data.valid[2], model, evaluateL2, evaluateL1,
                                               args.batch_size)
        val_loss=MSE
        train_loss_set.append(train_loss)

        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid_loss {:5.4f} | valid MAE {:5.4f} | valid RMSE  {:5.4f} | lr {:5.4f}'
            .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, MAE, RMSE,optim.lr))

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            MAE,MSE,RMSE= evaluate(Data, Data.test[0], Data.test[1], Data.test[2],model, evaluateL2, evaluateL1,
                                                     args.batch_size)
            print("test MAE {:5.4f} | test MSE {:5.4f} | test RMSE {:5.4f}".format(MAE,MSE,RMSE))

        optim.updateLearningRate(val_loss, epoch)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    model = torch.load(f)
MAE,MSE,RMSE = evaluate(Data, Data.test[0], Data.test[1], Data.test[2],model, evaluateL2, evaluateL1,
                                         args.batch_size)
print("test MAE {:5.4f} | test MSE {:5.4f} | test RMSE {:5.4f}".format(MAE, MSE,RMSE))
print("=="*30)


def prediction(model,data,X,Y_his,Y,inverse):
    model.eval()
    predict=None
    test = None

    for X, Y_his,Y in data.get_batches(X,Y_his, Y, args.batch_size, False):
        output = model(X,Y_his)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
    if inverse==1:
        scale = data.scale
    else:
        scale=1
    predict=predict*scale
    test=test*scale
    return predict,test

pre,test=prediction(model,Data,Data.test[0], Data.test[1],Data.test[2],args.inverse)
pre=pre.cuda().data.cpu()
test=test.cuda().data.cpu()

from sklearn.metrics import mean_squared_error,mean_absolute_error
pre=np.array(pre)
test=np.array(test)
print(f"MAE：{mean_absolute_error(pre, test)}")
print(f"MSE：{mean_squared_error(pre, test)}")
print(f"RMSE：{np.sqrt(mean_squared_error(pre, test))}")

np.save('./result/pred',pre)
np.save('./result/true',test)



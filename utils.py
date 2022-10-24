import numpy as np
import torch

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class Data_Utility(object):
    def __init__(self, raw_data, train, valid, divide_data,horizon, window, periods, ptl,cuda,dataset,pred_method):
        self.T = window
        self.cuda=cuda
        self.pred_method=pred_method
        if dataset=='traffic2':
            target=0
        else:
            target=-1
        self.raw_data = np.loadtxt(raw_data,delimiter=',')[:,target]
        self.horizon = horizon
        self.ptl = ptl
        self.periods = periods
        self.dat = np.zeros((self.raw_data.shape[0], sum(self.ptl) + 1))
        self.scale = 1
        self._normalized()
        self._ts_translation()
        self.cut = np.max(np.array(self.ptl) * (np.array(self.periods)))
        self.dat = self.dat[self.cut:, :]
        self.n, self.m = self.dat.shape
        if divide_data==1:
            self._split(train, train + valid)
        else:
            self._split(int(train * self.n), int((train + valid) * self.n))


    def _normalized(self):
        self.scale = np.max(np.abs(self.raw_data))
        self.dat[:, 0] = self.raw_data / self.scale

    def _ts_translation(self):                      #Seasonal Alignment
        dat = self.dat
        periods = self.periods
        ptl = self.ptl
        k = 1
        for i in range(len(ptl)):
            w = 1
            for j in range(ptl[i]):
                dat[periods[i]:, k] = dat[0:(-periods[i]), w - 1]
                k += 1
                w = k
        self.dat = dat

    def _split(self, train, valid):                 # Dividing datasets
        train_set = range(self.T + self.horizon - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        if self.pred_method==1:
            self.train = self._batchify1(train_set)
            self.valid = self._batchify1(valid_set)
            self.test = self._batchify1(test_set)
        else:
            self.train = self._batchify2(train_set)
            self.valid = self._batchify2(valid_set)
            self.test = self._batchify2(test_set)


    def _batchify1(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.T, self.m-1))  #Exogenous sequences other than Xo
        Y_his=torch.zeros((n,self.T-1))           #Xo. That is, the most recent observation
        Y = torch.zeros((n, 1))                          #Observations.  label
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.T
            X[i, :, :] = torch.from_numpy(self.dat[start:end, 1:])
            Y_his[i,:] = torch.from_numpy(self.dat[start:end-1, 0])
            Y[i,0] = self.dat[idx_set[i], 0]
        return [X, Y_his,Y]

    def _batchify2(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.T, self.m-1))
        Y_his=torch.zeros((n,self.T-1))
        Y = torch.zeros((n, self.horizon))
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.T
            X[i, :, :] = torch.from_numpy(self.dat[start:end, 1:])
            Y_his[i,:] = torch.from_numpy(self.dat[start:end-1, 0])
            Y[i,:] = torch.from_numpy(self.dat[end:end+self.horizon, 0])
        return [X, Y_his,Y]

    def get_batches(self, inputs,t_his, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y_his=t_his[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.cuda()
                Y_his = Y_his.cuda()
                Y = Y.cuda()
            yield X, Y_his,Y
            start_idx += batch_size
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as opt
import torch.utils.data
from torch.autograd import Variable

def conv(data):
    new = []
    for i in range(1,944):
        id_mvs = data[:,1][data[:,0] == i]
        id_rat = data[:,2][data[:,0] == i]
        rat = np.zeros(1682)
        rat[id_mvs - 1] = id_rat
        new.append(list(rat))
    return new

w = torch.randn(200,1682)
a = torch.randn(1,200)       
b = torch.randn(1,1682)

def hid(x):
    wx = torch.mm(x,w.t())
    atv = wx + a.expand_as(wx)
    ph = torch.sigmoid(atv)
    return ph, torch.bernoulli(ph)

def vis(y):
    wy = torch.mm(y,w)
    atv = wy + b.expand_as(wy)
    pv = torch.sigmoid(atv)
    return pv, torch.bernoulli(pv)
def train(v0, vk, ph0,phk):
    global w,a,b
    w += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(), phk)).t()
    b += torch.sum((v0 - vk), 0)
    a += torch.sum((ph0 - phk), 0)

trn = pd.read_csv('./u1.base',delimiter = '\t')
tst = pd.read_csv('./u1.test',delimiter = '\t')

trn = np.array(trn, dtype= 'int')
tst = np.array(tst, dtype= 'int')

print(max(max(trn[:,0]),max(tst[:,0])))
print(max(max(trn[:,1]),max(tst[:,1])))

trn = conv(trn)
tst = conv(tst)

trn = torch.FloatTensor(trn)
tst = torch.FloatTensor(tst)

trn[trn == 0] = -1
trn[trn == 1] = 0
trn[trn == 2] = 0
trn[trn >= 3] = 1
tst[tst == 0] = -1
tst[tst == 1] = 0
tst[tst == 2] = 0
tst[tst >= 3] = 1

for ep in range(1, 11):
    trnLss = 0
    s = 0.
    for i in range(0, 943, 100):
        vk = trn[i:i+100]
        v0 = trn[i:i+100]
        ph0,_ = hid(v0)
        for k in range(10):
            _,hk = hid(vk)
            _,vk = vis(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = hid(vk)
        train(v0, vk, ph0, phk)
        trnLss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(ep)+' loss '+str(trnLss/s))
tstLss = 0
s = 0.
for i in range(943):
    v = trn[i:i+1]
    vt = tst[i:i+1]
    if len(vt[vt>=0]) > 0:
        _,h = hid(v)
        _,v = vis(h)
        tstLss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(tstLss/s))

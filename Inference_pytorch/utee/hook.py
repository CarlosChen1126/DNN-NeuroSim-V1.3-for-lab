#from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
from modules.quantization_cpu_np_infer import QConv2d,QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer
import math

def Neural_Sim(self, input, output): 
    global model_n, FP

    print("quantize layer ", self.name)
    input_file_name =  './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(self.name) + '.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' ')
    if FP:
        weight_q = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
    else:
        #要改dnn-gating的話改這
        weight_q = wage_quantizer.Q(self.weight,self.wl_weight)
    new_weight = trans(weight_q.cpu().data.numpy(), 8)
    Lk = LRE(new_weight, 8)
    new_new_weight = LRE_index(new_weight, Lk, 8)
    #torch.set_printoptions(profile="full")
    #print(*new_weight)
    # with open("LRE_checking.txt", "w") as w:
    #     w.write(str(new_weight))
    #     w.write("======================================")
    #     w.write("======================================\n")
    #     w.write(str(Lk))
    #     w.write("======================================")
    #     w.write("======================================\n")
    #     w.write(str(new_new_weight))
    #     w.write("======================================")
    #     w.write("======================================")
    #     w.write("======================================")
    #     w.write("======================================")
    #     w.write("======================================\n")
    new_weight = booltoint(new_weight)
    new_new_weight = booltoint(new_new_weight)
    print(Lk)
    print("shape:::::\n")
    print(new_weight.shape)
    print(new_new_weight.shape)
    write_matrix_weight_m(new_weight, "test.csv")
    write_matrix_weight_m(new_new_weight, "test1.csv")
    

    # print(Lk)
    # print("======================================")
    # print("======================================")
    
    # print(*new_new_weight)
    # print("======================================")
    # print("======================================")
    # print("======================================")
    # print("======================================")
    # print("======================================")
    # print("======================================")
    write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)
    if len(self.weight.shape) > 2:
        k=self.weight.shape[-1]
        padding = self.padding
        stride = self.stride    
        write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride),None,self.wl_input,input_file_name)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)
def write_matrix_weight_m(input_matrix,filename):
    cout = input_matrix.shape[0]*8
    print(input_matrix.shape)
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')

def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    print(input_matrix.shape)
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')
def LRE(weight, bits):
    new_weight = np.empty((weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], bits))
    Lk = np.empty((weight.shape[2], weight.shape[3], weight.shape[1]))
    #m_weight用來記錄已被取用的row
    for i in range(int(weight.shape[2])):
        for j in range(int(weight.shape[3])):
            m_weight = np.zeros(weight.shape[1])
            mHD = np.empty((weight.shape[1],weight.shape[1]))
            #calculate mHD 
            for s in range(weight.shape[1]):
                for t in range(weight.shape[1]):
                    mHD[s][t] = 1000000
                    if s > t:
                        if(m_weight[s] + m_weight[t] == 0):
                            count = 0
                            for ii in range(weight.shape[0]):
                                for ind in range(8):
                                    if(int(weight[ii][s][i][j][ind]) + int(weight[ii][t][i][j][ind]) > 0):
                                        count+=1
                            mHD[s][t] = count
                            mHD[t][s] = count
                        
                        # 要想辦法寫如何把最小mHD的兩個
            # _min, _min_i, _min_j, m_weight = cal_minMHD(m_weight, mHD)
            #ToDo完成LRE algorithm ##
            for k in range(math.ceil(weight.shape[1]/8)):
                l = k*8
                _min, _min_i, _min_j, m_weight = cal_minMHD(m_weight, mHD)
                Lk[i][j][l] = _min_i
                Lk[i][j][l+1] = _min_j
                mask = np.empty((weight.shape[0] * 8))
                for x in range(weight.shape[0]):
                    for y in range(8):
                        ans = weight[x][_min_i][i][j][y] +weight[x][_min_j][i][j][y]
                        if(ans > 0):
                            mask[8*x+y] =1
                        else:
                            mask[8*x+y]=0
                l += 2
                while l < min((k+1)*8, weight.shape[1]):
                    mHD_m = np.empty((weight.shape[1]))
                    for s in range(weight.shape[1]):
                        mHD_m[s] = 1000000
                        if(m_weight[s] == 0):
                            count = 0
                            for ii in range(weight.shape[0]):
                                for ind in range(8):
                                    if(int(weight[ii][s][i][j][ind]) + mask[8*ii+ind] >0):
                                        count+=1
                            mHD_m[s] = count
                    if(all_eq_R(mHD_m, weight.shape[1]) == 1):
                        print("break")
                        break
                    _min_m, _min_i_m, m_weight = cal_minMHD_m(m_weight, mHD_m)
                    Lk[i][j][l] = _min_i_m
                    for x in range(weight.shape[0]):
                        for y in range(8):
                            ans = mask[8*x+y] + weight[x][_min_i_m][i][j][y]
                            if(ans > 0):
                                mask[8*x+y] =1
                            else:
                                mask[8*x+y]=0
                    l += 1
            
            for p in range(len(m_weight)):
                if(m_weight[p] == 0):
                    np.append(Lk[i][j],p)
                    m_weight[p] = 1
    return Lk
def cal_minMHD(m_weight, mHD):
    _min = 1000000
    _min_i = 0
    _min_j = 0
    for i in range(mHD.shape[0]):
        for j in range(mHD.shape[1]):
            if( i > j):
                if(m_weight[i] + m_weight[j] ==0):
                    if(_min > mHD[i][j]):
                        _min = mHD[i][j]
                        _min_i = i
                        _min_j = j
    m_weight[_min_i] = 1
    m_weight[_min_j] = 1
    return _min, _min_i, _min_j, m_weight

def cal_minMHD_m(m_weight, mHD):
    _min = 1000000
    _min_i = 0
    for i in range(len(mHD)):
        if(m_weight[i] == 0):
            if(_min > mHD[i]):
                _min = mHD[i]
                _min_i = i
    m_weight[_min_i] = 1
    return _min, _min_i, m_weight

def all_eq_R(mHD, R):
    index = 1
    for i in range(len(mHD)):
        if(mHD[i] != R):
            index = 0
    return index
def booltoint(weight):
    new_weight = np.empty((weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], 8))

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for k in range(weight.shape[2]):
                for l in range(weight.shape[3]):
                    for m in range(weight.shape[4]):
                        if(weight[i][j][k][l][m] == 0):
                            new_weight[i][j][k][l][m] = int(0)
                        else:
                            new_weight[i][j][k][l][m] = int(1)
    return new_weight
def trans(input_matrix, bits):
    weight = (input_matrix + 1) * (2**(bits-1))
    # weight.shape:
    #(16,3,3,3)
    #(N,D,K,K)
    #(16,16,3,3)
    #print(input_matrix.shape)
    new_weight = np.empty((weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], bits))
    for i in range(int(len(weight))):
        for j in range(int(len(weight[i]))):
            for k in range(int(len(weight[i][j]))):
                for l in range(int(len(weight[i][j][k]))):
                    binn = float2bin(weight[i][j][k][l], 8)
                    for m in range(bits):
                        new_weight[i][j][k][l][m] = int(binn[m])
    return new_weight
    #target = weight[0][0][0][0]
    #binn =float2bin(target, 8)
def LRE_index(weight, Lk, wbits):
    new_weight = np.empty((weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], wbits))
    
    for i in range(weight.shape[2]):
        for j in range(weight.shape[3]):
            for k in range(weight.shape[1]):
                index = int(Lk[i][j][k])
                print(index)
                
                for l in range(weight.shape[0]):
                    for m in range(wbits):
                        new_weight[l][k][i][j][m] = weight[l][index][i][j][m]
    return new_weight



#def OU_scan(weight, OUrow, OUcol):


def float2bin(target, bits):
    new_bin = []
    for i in range(bits-1, -1, -1):
        new_bin.append(int(target // (2**i)))
        target %= (2**i)
    #print(new_bin)
    return new_bin

def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    print('input_shape = {}'.format(input_shape))

    #item_num = ((input_shape[2] + 2*padding[0] - window_size) / stride[0] + 1) * ((input_shape[3] + 2*padding[1] - window_size) / stride[1] + 1)
    item_num = int((input_shape[2] + 2*padding[0] - window_size) / stride[0] + 1) * int((input_shape[3] + 2*padding[1] - window_size) / stride[1] + 1)
   # print('item_num = {}'.format(item_num))
    output_matrix = np.zeros((input_shape[0],int(item_num),input_shape[1]*window_size*window_size))
    iter = 0
    for i in range(0, input_shape[2]-window_size + 1, stride[0]):
        for j in range(0, input_shape[3]-window_size + 1, stride[1]):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1
    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model,wl_weight,wl_activation,model_name,mode): 
    global model_n, FP
    model_n = model_name
    FP = 1 if mode=='FP' else 0
    
    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(model_name)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' ')
    
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (FConv2d, QConv2d, nn.Conv2d)) or isinstance(layer, (FLinear, QLinear, nn.Linear)):
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list

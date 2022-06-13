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
    """
    weight = (weight_q + 1) * ((2**7)-1)
    #weight = torch.clamp(weight_q,0 ,127)
    int_weight2D = write_matrix_weight_mm(weight, './layer_record_modified' + str(model_n) + '/weight' + str(self.name) + '_modified_'+'.csv')
    new_weight = trans(int_weight2D, 8)
    Lk = LRE_2D(new_weight, 8)
    #(16*8,27)
    #new_weight_2D = write_matrix_weight_m(new_weight, './layer_record_modified' + str(model_n) + '/weight' + str(self.name) + '_cellBit1_'+'.csv')
    #Lk = LRE(new_weight, 8)
    new_new_weight = LRE_index(new_weight, Lk, 8)
    #new_weight = booltoint(new_weight, 8)
    #new_new_weight = booltoint(new_new_weight)
    # weight = (weight_q + 1) * ((2**7)-1)
    # #weight = torch.clamp(weight_q,0 ,127)
    # write_matrix_weight_mm(weight, './layer_record_modified' + str(model_n) + '/weight' + str(self.name) + '_modified_'+'.csv')
    
    new_weight = write_matrix_weight_m(new_weight, './layer_record_modified' + str(model_n) + '/weight' + str(self.name) + '_cellBit1_'+'.csv')
    new_new_weight = write_matrix_weight_mmm(new_new_weight, './layer_record_modified' + str(model_n) + '/weight' + str(self.name) + '_after_LRE'+'.csv')
    print("2D_weight_matrix_shape:")
    print(new_weight.shape)
    print("===========================================")
    #(16*8, 27)
    rws_compressed_col = []
    lre_compressed_col = []
    ris_compressed_col = []

    lre_compressed_col = LRE_compression(new_new_weight.transpose(), 8, 8)
    rws_compressed_col = RWS_compression(new_new_weight.transpose(), 8, 8)

    write_matrix_weight(weight_q.cpu().data.numpy(),weight_file_name)
    if len(self.weight.shape) > 2:
        k=self.weight.shape[-1]
        padding = self.padding
        stride = self.stride
        activation_2D = write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(),k,padding,stride),None,self.wl_input,input_file_name)
        ris_compressed_col = RIS_compression(activation_2D,8,8)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)
    cycle = cal_OUcycle(activation_2D, new_weight, 8, 8, [], [], [])
    new_cycle = cal_OUcycle(activation_2D, new_weight, 8, 8, lre_compressed_col, rws_compressed_col, ris_compressed_col)
    print("===================================================")
    print("layer ", str(self.name))
    print("MVM_cycle(original) : ", cycle)
    print("Use the compressed algorithm:")
    if(lre_compressed_col != []):
        print("LRE")
    if(rws_compressed_col != []):
        print("RWS")
    if(ris_compressed_col != []):
        print("RIS")
    print("MVM_cycle(after_compressed) : ", new_cycle)
    print("compressed_ratio : ", (cycle-new_cycle)*100/cycle, "%")
    print("===================================================")
    with open("VGG8_experiment(LRE).txt", "a") as file:
        file.write("layer " + str(self.name)+"\n")
        file.write("MVM_cycle(original) : "+ str(cycle)+"\n")
        file.write("Use the compressed algorithm:")
        if(lre_compressed_col != []):
            file.write(" LRE ")
        if(rws_compressed_col != []):
            file.write(" RWS ")
        if(ris_compressed_col != []):
            file.write(" RIS ")
        file.write("\n"+"MVM_cycle(after_compressed) : " + str(new_cycle)+"\n")
        file.write("compressed_ratio : "+ str((cycle-new_cycle)*100/cycle) +"%\n")
        file.write("===================================================\n")
#0 1 format(cell format)
def write_matrix_weight_m(input_matrix,filename):
    input_matrix = input_matrix.transpose()
    np.savetxt(filename, input_matrix, delimiter=",",fmt='%d')
    return input_matrix.transpose()
#after LRE format
def write_matrix_weight_mmm(input_matrix,filename):
    input_matrix = input_matrix.transpose()
    np.savetxt(filename, input_matrix, delimiter=",",fmt='%d')
    return input_matrix.transpose()
#int format
def write_matrix_weight_mm(input_matrix,filename):
    #(16,3,3,3)
    D = input_matrix.shape[2] #3
    new_weight_matrix = np.empty((input_matrix.shape[1]*input_matrix.shape[2]*input_matrix.shape[3],input_matrix.shape[0]))
    for k in range(input_matrix.shape[1]):
        for i in range(input_matrix.shape[2]):
            for j in range(input_matrix.shape[3]):
                cout_y = D*D*k+D*i+j
                for l in range(input_matrix.shape[0]):
                    new_weight_matrix[cout_y][l] = input_matrix[l][k][i][j]
    np.savetxt(filename, new_weight_matrix, delimiter=",",fmt='%d')
    return new_weight_matrix
def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')
def LRE_2D(weight, bits):
    R = weight.shape[1]
    R = int(R)
    new_weight = np.empty((weight.shape[0], weight.shape[1]))
    Lk = np.empty((math.floor(R/bits), bits)) #LRE index
    m_weight = np.zeros((R)) #used rows
    mHD = np.empty((R, R))
    for i in range(R):
        for j in range(R):
            if i==j:
                mHD[i][j] = 0
            elif i>j:
                #calculate mHD#
                count = 0
                for k in range(weight.shape[0]):
                    if(weight[k][i] + weight[k][j] > 0):
                        count += 1
                mHD[i][j] = count
                mHD[j][i] = count
    for i in range(0, math.floor((R-1)/bits)):
        #j = i * bits
        j = 0
        _min, _min_i, _min_j, m_weight = cal_minMHD(m_weight, mHD)
        Lk[i][j] = _min_i
        Lk[i][j+1] = _min_j
        mask = np.empty((weight.shape[0]))
        for s in range(weight.shape[0]):
            if(weight[s][_min_i] + weight[s][_min_j] >0):
                mask[s] = 1
            else:
                mask[s] = 0
        j +=2
        while (j<8): 
            mHD_m = np.empty((R))
            for row in range(R):
                count = 0
                for iii in range(weight.shape[0]):
                    if(m_weight[row] == 0):
                        if(weight[iii][row] + mask[iii] > 0 ):
                            count+=1
                mHD_m[row] = count
            if(all_eq_R(mHD_m, 128) == 1):
                print("break")
                break
            _min_m, _min_i_m, m_weight = cal_minMHD_m(m_weight, mHD_m)
            Lk[i][j] = _min_i_m
            for ii in range(weight.shape[0]):
                if(weight[ii][_min_i_m] + mask[ii] >0):
                    mask[ii] = 1
                else:
                    mask[ii] = 0
            j += 1
    Lk_new = np.empty((R))
    for i in range(math.floor(R/bits)):
        for j in range(bits):
            Lk_new[8*i+j] = Lk[i][j]
    start = bits*math.floor((R-1)/bits)
    for i in range(R):
        if(m_weight[i] == 0):
            Lk_new[start] = i
            start+=1
    return Lk_new

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
def booltoint(weight, bits):
    new_weight = np.empty((weight.shape[0], weight.shape[1]))

    for x in range(weight.shape[0]):
        for y in range(weight.shape[1]):
            if(weight[x][y] != 0):
                new_weight[x][y] = int(1)
            else:
                new_weight[x][y] = int(0)
    return new_weight
def trans(input_matrix, bits):
    # weight = (input_matrix + 1) * (2**(int(bits)-1))
    # D = input_matrix.shape[2]
    # new_weight_matrix = np.empty((input_matrix.shape[0], input_matrix.shape[1]*input_matrix.shape[2]*input_matrix.shape[3]))
    # for k in range(input_matrix.shape[1]):
    #     for i in range(input_matrix.shape[2]):
    #         for j in range(input_matrix.shape[3]):
    #             cout_y = D*D*k+D*i+j
    #             for l in range(input_matrix.shape[0]):
    #                 new_weight_matrix[l][cout_y] = weight[l][k][i][j]
    # weight.shape:
    #(N,D,K,K)
    #(16,3,3,3)
    #(16,16,3,3)
    #(32,16,3,3)
    #(32,32,3,3)
    new_weight_matrix = input_matrix.transpose()
    new_weight = np.empty((new_weight_matrix.shape[0]*int(bits), new_weight_matrix.shape[1]))
    for x in range(new_weight_matrix.shape[0]):
        for y in range(new_weight_matrix.shape[1]):
            temp = float2bin(new_weight_matrix[x][y], bits)
            C = 8*x
            for m in range(bits):
                new_weight[C+m][y] = temp[m]
    return new_weight
    #target = weight[0][0][0][0]
    #binn =float2bin(target, 8)
def LRE_index(weight, Lk, wbits):
    new_weight = np.empty((weight.shape[0], weight.shape[1]))    
    for i in range(weight.shape[1]):
        index = int(Lk[i])
        for j in range(weight.shape[0]):
            new_weight[j][i] = weight[j][index]    
    return new_weight

def LRE_compression(weight, OUrow, OUcol):
    #(27,128)
    #print("weight.shape: ", weight.shape)
    compressed_col = 0 #OU size col
    compressed_col_eachOU = np.zeros((math.ceil(weight.shape[0]/OUrow)))
    for i in range(math.floor(weight.shape[0]/OUrow)):
        for j in range(math.floor(weight.shape[1]/OUcol)):
            ou_matrix = np.empty((OUrow, OUcol))
            for k in range(OUrow):
                for l in range(OUcol):
                    ind_x = (OUrow*i) +k
                    ind_y = (OUcol*j) +l
                    #ind_x = min(ind_x, weight.shape[0]-1)
                    #ind_y = min(ind_y, weight.shape[1]-1)
                    #print("ind_x")
                    #print(ind_x)
                    ou_matrix[k][l] = weight[ind_x][ind_y] #OU of now's MVM
            for m in range(OUcol):
                verify = 0
                for n in range(OUrow):
                    verify += ou_matrix[n][m]
                    #print(ou_matrix[n][m])
                if(verify == 0):
                    compressed_col += 1
                    compressed_col_eachOU[i] += 1
                    #print("Compressed!")
    x = math.floor(weight.shape[0]/OUrow)
    
    if(weight.shape[0]-x*OUrow > 0):
        small_compressed_col=0
        for i in range(weight.shape[1]):
            verify=0
            for j in range(x*OUrow, weight.shape[0]):
                verify+=weight[j][i]
            if(verify ==0):
                small_compressed_col+=1
                #print("small_compressed")
    print("===========================================")
    print("LWS")
    print("compressed_col(OUcol_size): ",compressed_col)
    if(weight.shape[0]-x*OUrow > 0):
        compressed_col_eachOU[math.ceil(weight.shape[0]/OUrow)-1] = small_compressed_col
        print("rest_compressed_col(col): ",small_compressed_col)
    print("compressed_col_eachOU: ",compressed_col_eachOU)
    print("===========================================")
    return compressed_col_eachOU

def RWS_compression(weight, OUrow, OUcol):
    compressed_col = 0
    compressed_col_eachOU = np.zeros((math.ceil(weight.shape[0]/OUrow)))
    x=math.floor(weight.shape[0]/OUrow)
    y=math.floor(weight.shape[1]/OUcol)
    #record = np.empty(2**OUrow)
    record = np.zeros((x,2**OUrow))
    record_rep = np.zeros((x,2**OUrow))
    record_repp = np.zeros((x,2**OUrow))
    # print("weight.shape")
    # print(x)
    # print(y)
    for i in range(math.floor(weight.shape[0]/OUrow)):
        for j in range(math.floor(weight.shape[1]/OUcol)):
            ou_matrix = np.empty((OUrow, OUcol))
            for k in range(OUrow):
                for l in range(OUcol):
                    ind_x = (OUrow*i) +k
                    ind_y = (OUcol*j) +l
                    ou_matrix[k][l] = weight[ind_x][ind_y]
            
            for m in range(OUcol):
                verify = 0
                col_weight=0
                for n in range(OUrow):
                    col_weight+=ou_matrix[m][n]*(2**n)
                if(col_weight >0):
                    if(col_weight > 255):
                        print("something error qq")
                    if(record[i][int(col_weight)]==1):
                        record_rep[i][int(col_weight)] += 1
                        record_repp[i][int(col_weight)] += 1
                        compressed_col+=1
                        compressed_col_eachOU[i]+=1
                        #print("compressed!")
                    else:
                        record[i][int(col_weight)]=1
                        record_rep[i][int(col_weight)] += 1
        print("compressed_col_each: ",compressed_col_eachOU[i])
        print(record_rep[i])
        print(record_repp[i])
        print(record_rep[i].sum())
        print(record_repp[i].sum())
    x = math.floor(weight.shape[0]/OUrow)
    num = weight.shape[0] -x*OUrow
    if(num > 0):
        record_rest = np.zeros((2**num))
        small_compressed_col=0
        for i in range(weight.shape[1]):
            verify=0
            col_weight=0
            for j in range(x*OUrow, weight.shape[0]):
                n = weight.shape[0]-j-1
                col_weight += weight[j][i]*(2**n)
            if(col_weight > 0):
                col_weight = int(col_weight)
                if (record_rest[col_weight] == 0):
                    record_rest[col_weight] = 1
                else:
                    small_compressed_col +=1
    print("===========================================")
    print("RWS")
    print("compressed_col: ",compressed_col)
    if(num > 0):
        compressed_col_eachOU[math.ceil(weight.shape[0]/OUrow)-1] = small_compressed_col
        print("rest_compressed_col(col): ",small_compressed_col)
    print("compressed_col_eachOU: ",compressed_col_eachOU)
    print("===========================================")
    return compressed_col_eachOU
#
def RIS_compression(activation, OUrow, OUcol):######不同OU間重複的要分開考慮######
    print("ac.shape ", activation.shape)
    # print(activation.shape)
    x = math.floor(activation.shape[0]/OUrow)
    y = math.floor(activation.shape[1]/OUcol)
    compressed_input = 0
    compressed_input_eachOU = np.zeros(math.ceil(activation.shape[0]/OUcol))
    record = np.zeros((x,y,2**OUrow))
    #(27, 8192)
    #(144, 8192)
    for i in range(math.floor(activation.shape[1]/OUrow)):
        for j in range(math.floor(activation.shape[0]/OUcol)):
            for k in range(OUcol):
                col_activation = 0
                for l in range(OUrow):
                    ind_x = j* OUcol +k
                    ind_y = i*OUrow + l
                    col_activation += int(activation[ind_x][ind_y])*(2**l)
                if(record[j][i][int(col_activation)] == 1):
                    compressed_input +=1
                    compressed_input_eachOU[j] += 1

                else:
                    record[j][i][int(col_activation)] = 1
                    #print("activation compressed")
    print("===========================================")
    print("RIS")
    print("compressed_activations: ",compressed_input)
    num = activation.shape[0] -x*OUrow
    if(num > 0):
        record_rest = np.zeros((y, 2**num))
        small_compressed_col = 0
        for i in range(math.floor(activation.shape[1]/OUrow)):
            for j in range(OUrow):
                col_activation=0
                for k in range(num):
                    ind_x = OUrow*i + j 
                    ind_y = x*OUrow + k
                    col_activation += int(activation[ind_y][ind_x])*(2**k)
                if (record_rest[i][int(col_activation)] == 1):
                    small_compressed_col += 1
                else:
                    record_rest[i][int(col_activation)] = 1
        compressed_input_eachOU[math.ceil(activation.shape[0]/OUcol)-1] = small_compressed_col
        print("rest_compressed_col(col): ",small_compressed_col)
    print("compressed_input_eachOU: ",compressed_input_eachOU)
    print("===========================================")


    return compressed_input_eachOU

def float2bin(target, bits):
    new_bin = np.empty(bits)
    for i in range(bits):
        val = target // (2**(bits-1-i))
        new_bin[i] = val
        target = target % (2**(bits-1-i))
    return new_bin
def cal_OUcycle(input, weight, OUrow, OUcol, lre_compressed_col, rws_compressed_col, ris_compressed_col):
    #(27,8192)
    #(128,27)
    input = input.transpose()
    #(8192,27)
    #(128,27)
    ax = input.shape[0]
    ay = input.shape[1]
    wx = weight.shape[0]
    wy = weight.shape[1]
    totalW = np.zeros(math.ceil(weight.shape[1]/OUrow))
    totalA = np.zeros(math.ceil(weight.shape[1]/OUrow))
    final = 0
    if(lre_compressed_col == []):
        lre_compressed_col = np.zeros(math.ceil(weight.shape[1]/OUrow))
    if(rws_compressed_col == []):
        rws_compressed_col = np.zeros(math.ceil(weight.shape[1]/OUrow))
    if(ris_compressed_col == []):
        ris_compressed_col = np.zeros(math.ceil(weight.shape[1]/OUrow))

    for i in range(math.ceil(weight.shape[1]/OUrow)):
        totalW[i] = wx - lre_compressed_col[i] - rws_compressed_col[i]
        totalA[i] = ax - ris_compressed_col[i]
        final += (totalW[i]*totalA[i])
    return final
"""
def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    #(27,1024*8=8192)
    #(144,1024*8=8192)
    print("-----------------: ",input_matrix.shape)
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    print(filled_matrix_b.shape)
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')
    return filled_matrix_b


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')

#input size:W*W*D
#W=32; D=27,144

def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    print('input_shape = {}'.format(input_shape))

    #item_num = ((input_shape[2] + 2*padding[0] - window_size) / stride[0] + 1) * ((input_shape[3] + 2*padding[1] - window_size) / stride[1] + 1)
    item_num = int((input_shape[2] + 2*padding[0] - window_size) / stride[0] + 1) * int((input_shape[3] + 2*padding[1] - window_size) / stride[1] + 1)
    print('item_num = {}'.format(item_num))
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

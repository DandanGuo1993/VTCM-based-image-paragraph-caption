import numpy as np
import numpy.ctypeslib as npct
import ctypes
from ctypes import *




array_2d_double = npct.ndpointer(dtype=np.double,ndim=2,flags='C')
array_1d_double = npct.ndpointer(dtype=np.double,ndim=1,flags='C')
array_int = npct.ndpointer(dtype=np.int32,ndim=0,flags='C')
ll = ctypes.cdll.LoadLibrary   

Multi_lib = ll("./libMulti_Sample.dll")
Multi_lib.Multi_Sample.restype = None
Multi_lib.Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]
Multi_lib.Multi_Input.restype = None
Multi_lib.Multi_Input.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int]

Crt_Multi_lib = ll("./libCrt_Multi_Sample.dll")
Crt_Multi_lib.Crt_Multi_Sample.restype = None
Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double,array_2d_double,array_2d_double,array_2d_double,array_2d_double, c_int, c_int, c_int]

Crt_lib = ll("./libCrt_Sample.dll")
Crt_lib.Crt_Sample.restype = None
Crt_lib.Crt_Sample.argtypes = [array_2d_double,array_2d_double, array_2d_double, c_int,c_int]



realmin = 2.2e-308

def Calculate_pj(c_j,T):  
    p_j = []
    N = c_j[1].size
    p_j.append((1-np.exp(-1))*np.ones([1,N]))     
    p_j.append(1/(1+c_j[1]))
    
    for t in [i for i in range(T+1) if i>1]:    
        tmp = -np.log(np.maximum(1-p_j[t-1],realmin))
        p_j.append(tmp/(tmp+c_j[t]))
        
    return p_j


def Multrnd_Matrix(X_t,Phi_t,Theta_t):

    V = X_t.shape[0]
    J = X_t.shape[1]
    K = Theta_t.shape[0]
    Xt_to_t1_t = np.zeros([K,J], order = 'C').astype('double')
    WSZS_t = np.zeros([V,K], order = 'C').astype('double')
    
    Multi_lib.Multi_Sample(X_t,Phi_t,Theta_t, WSZS_t, Xt_to_t1_t, V,K,J)

    return Xt_to_t1_t, WSZS_t   

def Multrnd_Input(X_t,Phi_1,Theta_1,Phi_2,Theta_2,k_1,k_2):
    V = X_t.shape[0]
    J = X_t.shape[1]

  
    Pro1 = np.dot(Phi_1,Theta_1).astype('double') / np.maximum( (np.sqrt( np.dot(Phi_1,Theta_1) * np.dot(Phi_1,Theta_1) ).sum(0)),1 ) 
    Pro1 = Pro1 * k_1
        
    Pro2 = np.dot(Phi_2,Theta_2).astype('double') / np.maximum( (np.sqrt( np.dot(Phi_2,Theta_2) * np.dot(Phi_2,Theta_2) ).sum(0)),1 )

    Pro2 = Pro2 * k_2
    X_t_1 = np.zeros([V,J], order = 'C').astype('double')
    X_t_2 = np.zeros([V,J], order = 'C').astype('double')
    
    Multi_lib.Multi_Input(X_t.astype('double'),Pro1,Pro2,X_t_1,X_t_2,V,J)
    return X_t_1,X_t_2
    
def Crt_Multirnd_Matrix(Xt_to_t1_t,Phi_t1,Theta_t1):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Kt1 = Theta_t1.shape[0]
    Xt_to_t1_t1 = np.zeros([Kt1,J],order = 'C').astype('double')
    WSZS_t1 = np.zeros([Kt,Kt1],order = 'C').astype('double')
    
    Crt_Multi_lib.Crt_Multi_Sample(Xt_to_t1_t, Phi_t1,Theta_t1, WSZS_t1, Xt_to_t1_t1, Kt, Kt1 , J)
    
    
    return Xt_to_t1_t1 , WSZS_t1

def Crt_Matrix(Xt_to_t1_t, p ):
    Kt = Xt_to_t1_t.shape[0]
    J = Xt_to_t1_t.shape[1]
    Xt_t1 = np.zeros([Kt,J],order = 'C').astype('double')
    
    Crt_lib.Crt_Sample(Xt_to_t1_t, p.astype('double'), Xt_t1, Kt, J)
    
   
    return Xt_t1

    
def Sample_Phi(WSZS_t,Eta_t):   
    Kt = WSZS_t.shape[0]
    Kt1 = WSZS_t.shape[1]
    Phi_t_shape = WSZS_t + Eta_t
    Phi_t = np.zeros([Kt,Kt1])
    Phi_t = np.random.gamma(Phi_t_shape,1)

    Phi_t = Phi_t / Phi_t.sum(0)
    return Phi_t
    
def Sample_Theta(Xt_to_t1_t,c_j_t1,p_j_t,shape):
    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    Theta_t = np.zeros([Kt,N])
    Theta_t_shape = Xt_to_t1_t + shape
    Theta_t[:,:] = np.random.gamma(Theta_t_shape,1) / (c_j_t1[0,:]-np.log(np.maximum(realmin,1-p_j_t[0,:])))

    return Theta_t


def Sample_Theta_2(Xt_to_t1_t,shape):
    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    Theta_t = np.zeros([Kt,N])
    Theta_t_shape = Xt_to_t1_t + shape
    Theta_t[:,:] = np.random.gamma(Theta_t_shape,1) 

    return Theta_t

def ProjSimplexSpecial(Phi_tmp,Phi_old,epsilon):
    Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
    if  np.where(Phinew[:,:]<=0)[0].size >0:
        Phinew = np.maximum(epsilon,Phinew)
        Phinew = Phinew/np.maximum(realmin,Phinew.sum(0))
    return Phinew

def Reconstruct_error(X,Phi,Theta):
    return np.power(X-np.dot(Phi,Theta),2).sum()


def MultRnd(value, MultRate):
    MultRate = MultRate / np.sum(MultRate)
    MultRate_Sum = np.reshape(MultRate, [-1])
    N = len(MultRate_Sum)
    Amnk = np.zeros([N])

    if N == 1:
        Amnk = value
    else:
        for i in range(1, N, 1):  
            MultRate_Sum[i] = MultRate_Sum[i] + MultRate_Sum[i - 1]

        Uni_Rnd = np.random.rand(np.int64(value))  
        flag_new = Uni_Rnd <= MultRate_Sum[0]
        Amnk[0] = np.sum(flag_new)

        for i in range(1, N, 1):  
            flag_old = flag_new
            flag_new = Uni_Rnd <= MultRate_Sum[i]
            Amnk[i] = np.sum(~flag_old & flag_new)

        Amnk = np.reshape(Amnk, MultRate.shape)

    return Amnk


def Dis_Dic(D):
    [K, K1, K2] = D.shape
    w_n = np.ceil(np.sqrt(K))
    h_n = np.ceil(K / w_n)
    weight = w_n * K2
    height = h_n * K1
    Dic = np.zeros([np.int32(weight), np.int32(height)])
    count = 0
    for k1 in range(np.int32(w_n)):
        for k2 in range(np.int32(h_n)):
            Dic[k1 * K1: (k1 + 1) * K1, k2 * K2: (k2 + 1) * K2] = D[count, :, :]
            count += 1
            if count == K:
                break
        if count == K:
            break



def Conv_Aug(Kernel, Score_Shape):
    [K1, K2] = Score_Shape
    [K3, K4] = Kernel.shape
    V1 = K1 + K3 - 1
    V2 = K2 + K4 - 1


    Kernel_Pad = np.zeros([2 * V1 - K3, 2 * V2 - K4])  
    Kernel_Pad[V1 - K3: V1, V2 - K4: V2] = Kernel
    Kernel_Pad = Kernel_Pad.T
    M, N = Kernel_Pad.shape
    col_extent = N - K1 + 1
    row_extent = M - K2 + 1

    start_idx = np.arange(K2)[:, None] * N + np.arange(K1)
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(Kernel_Pad, start_idx.ravel()[:, None] + offset_idx.ravel())

    return np.flip(out.T, axis=1)
    
    

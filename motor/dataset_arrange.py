import numpy as np
import cupy as cp

# 讀取 txt 文件
def loadSimData(path_x, path_p):
    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, z_data, x_k_predict_data
    x_k_update_data = x_data[:, 0:2]
    k_y_data = x_data[:, 2:4]
    x_tel = x_data[:, 4:6]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 6:8] # x_true_data
    x_true_noise = x_data[:, 8:10] # x_true_data_noise
    x_obsve = x_data[:, 10]# z_data
    x_k_predict_data = x_data[:, 11:13]
    x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:4]
    KCP_data = P_data[:, 4:8]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all

def loadMotorData(path_x, path_P):
    # 馬達實際資料
    x_data = np.loadtxt(path_x) # motor_dataset/Motor_x_data.txt
    P_data = np.loadtxt(path_P) # motor_dataset/Motor_P_data.txt
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
    # Pos, Vel, Acc_LAE, pose, vele, acce, PosCmd, VelCmd, AccCmd, km_y_data
    x_true = x_data[:,:1]
    # print("x_true.shape =", x_true.shape)

    # x_k_update_data2 = []
    # for i in range(14681):
    #     x_k_update = np.concatenate((x_data[i:i+3, 3].reshape(1, 3), x_data[i:i+3, 4].reshape(1, 3), x_data[i:i+3, 5].reshape(1, 3)), axis = 1)
    #     x_k_update_data2.append(x_k_update)
    # x_k_update_data2 = np.squeeze(x_k_update_data2)

    # x_k_update_data2 = np.array(x_k_update_data2).reshape(np.array(x_k_update_data2).shape[0], -1)
    
    # x_k_update_data = x_data[:, 3:6]
    # x_cmd = x_data[:, 6:9]
    # km_y_data = x_data[:14681, 9:]
    # x_tel = x_cmd[:14681, :] - x_k_update_data[:14681, :]

    x_k_update_data = x_data[:, 3:6]
    x_cmd = x_data[:, 6:9]
    km_y_data = x_data[:, 9:]
    x_tel = x_cmd - x_k_update_data
   
<<<<<<< HEAD
    # x_input_data_all = np.concatenate((x_true, x_k_update_data, km_y_data, x_tel), axis = 1)
    # x_input_data_all = np.concatenate((x_k_update_data, km_y_data, x_tel), axis = 1)
    # x_input_data_all = np.concatenate((x_true, x_true, x_true, x_cmd), axis = 1)
=======
    # x_input_data_all = np.concatenate((x_k_update_data, km_y_data, x_tel), axis = 1)
    x_input_data_all = np.concatenate((x_true, x_true, x_true, x_cmd), axis = 1)
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
    # x_input_data_all = np.concatenate((x_k_update_data, x_true, x_cmd), axis = 1)
    # x_input_data_all = np.concatenate((x_true, x_true, x_true, x_true, x_cmd), axis = 1)

    # x_input_data_all = np.concatenate((x_true, x_true, x_true, km_y_data, x_tel), axis = 1)
<<<<<<< HEAD
    x_input_data_all = np.concatenate((x_true, x_k_update_data, x_cmd), axis = 1)
    # x_input_data_all = np.concatenate((x_true, x_true, x_true), axis = 1)
    # x_input_data_all = x_k_update_data
=======
    # x_input_data_all = np.concatenate((x_true, x_true, x_true), axis = 1)
    # x_input_data_all = x_k_update_data
=======
    # Pos, pose, vele, acce, PosCmd, VelCmd, AccCmd, km_y_data
    x_true = x_data[:,0]
    x_k_update_data = x_data[:, 1:4]
    x_cmd = x_data[:, 4:7]
    km_y_data = x_data[:, 7:8]
    x_tel = x_cmd - x_k_update_data
    # km_y_data = np.array(km_y_data).reshape(-1, 1)
    # x_tel = np.array(x_tel)
    # print("x_k_update_data.shape =", x_k_update_data.shape)
    # print("km_y_data.shape =", km_y_data.shape)
    # print("x_tel.shape =", x_tel.shape)
    x_input_data_all = np.concatenate((x_k_update_data, km_y_data, x_tel), axis = 1)
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
    # Pm_data, kcp_data
    P_k_update_data = P_data[:, :9]
    KCP_data = P_data[:, 9:]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
<<<<<<< HEAD
    # print("x_ture =", x_true)
    
=======
<<<<<<< HEAD
    # print("x_ture =", x_true)
    
=======
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
>>>>>>> 6a91b5d423c756b82df34fa1d19ee44af9e1ac77
    return x_data, x_true, x_k_update_data, x_cmd, km_y_data, x_tel, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all
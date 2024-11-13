import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
<<<<<<< HEAD
import setLSTMConfig
=======
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 訓練參數設置
epoch = 300
<<<<<<< HEAD
traning_size = 23000   # diff: 14683 ；diff_2: 29366 # same: 23000
batch_size = 1000
data_set_size = traning_size

setConfig = setLSTMConfig.LSTMConfig()
x_input_size, x_output_size, hidden_size, num_layers, dropout, P_input_size, P_output_size = setConfig.getLSTMConfig()
# x初始化LSTM模型
x_lstm_model = LSTM.LSTM_KF(x_input_size, hidden_size, x_output_size, num_layers, dropout)
x_lstm_model = x_lstm_model.to(device)
x_optimizer = torch.optim.Adam(x_lstm_model.parameters(), lr=0.001)
x_loss_fn = nn.MSELoss()

# P初始化LSTM模型
P_lstm_model = LSTM.LSTM_KF(P_input_size, hidden_size, P_output_size, num_layers, dropout)
P_lstm_model = P_lstm_model.to(device)
P_optimizer = torch.optim.Adam(P_lstm_model.parameters(), lr=0.001)
P_loss_fn = nn.MSELoss()

# 馬達實際資料
path1 = 'motor_dataset/Motor_x_data_combine_diff_2.txt'
path2 = 'motor_dataset/Motor_P_data_combine_diff_2.txt'
x_data, x_true, x_k_update_data, x_cmd, km_y_data, x_tel, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadMotorData(path1, path2)

P_true = cp.array([[1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7]])
# P_true = cp.array([[1e-7]])
=======
traning_size = 10000
batch_size = 2500
hidden_size = 128
data_set_size = traning_size


## 讀取檔案 "D:\ASUS_program_code\規格測試\有線\IPS650_G50_motion.txt"
# path1 = ['E:/cmake_mouse_boundary_v9_1/build/IPS750_G50_F_motion.txt'] #馬達資料.txt路徑
# path2 = ['E:/cmake_mouse_boundary_v9_1/build/IPS750_G50_F_mouse.txt']  #滑鼠資料.txt路徑
# x_kf_update_data, P_kf_update_data, K_update_data, k_y_update_data, KCP_data, H, Pos, PosCmd, VelCmd, AccCmd, PosCmd_AddNoise, VelCmd_AddNoise, AccCmd_AddNoise = KF.KF_Process(path1, path2)
# x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data, prediction_errors_data = KF_training_data.KF_Process(traning_size)    

# x初始化LSTM模型
input_size = 7 # 包含狀態預測和卡爾曼增益等輸入特徵
output_size = 3  # 輸出狀態估計
x_lstm_model = LSTM.LSTM_KF(input_size, hidden_size, output_size)
# 將模型移動到 GPU（如果可用）
x_lstm_model = x_lstm_model.to(device)
# 損失函數和優化器
x_optimizer = torch.optim.Adam(x_lstm_model.parameters(), lr=0.0001)
x_loss_fn = nn.MSELoss()

# P初始化LSTM模型
input_size = 18  # 包含狀態預測和卡爾曼增益等輸入特徵
output_size = 9  # 輸出狀態估計
P_lstm_model = LSTM.LSTM_KF(input_size, hidden_size, output_size)
# 將模型移動到 GPU（如果可用）
P_lstm_model = P_lstm_model.to(device)
# 損失函數和優化器
P_optimizer = torch.optim.Adam(P_lstm_model.parameters(), lr=0.0001)
P_loss_fn = nn.MSELoss()

# 輸入模擬資料
# x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadSimData('sim_dataset/x_data_all_15000.txt', 'sim_dataset/P_data_all_15000.txt')
# x_input_data_all = np.loadtxt('x_input_data_all_normalized.txt', delimiter=' ')
# P_input_data_all = np.loadtxt('P_input_data_all_normalized.txt', delimiter=' ')
# 馬達實際資料
path1 = 'motor_dataset/Motor_x_data_ips300_cycle200.txt'
path2 = 'motor_dataset/Motor_P_data_ips300_cycle200.txt'
x_data, x_true, x_k_update_data, x_cmd, km_y_data, x_tel, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadMotorData(path1, path2)
# x_data = np.loadtxt('motor_dataset/Motor_x_data.txt')
# P_data = np.loadtxt('motor_dataset/Motor_P_data.txt')
# # Pos, pose, vele, acce, km_y_data, x_tel_data
# x_input_data_all = x_data[:, 1:]
# x_true = x_data[:,0]
# x_k_update_data = x_data[:, 1:4]
# # Pm_data, kcp_data
# P_input_data_all = P_data
# P_k_update_data = P_data[:, :9]
# KCP_data = P_data[:, 9:17]

# P_true = cp.array([[1e-7, 1e-7, 1e-7],
#                    [1e-7, 1e-7, 1e-7],
#                    [1e-7, 1e-7, 1e-7]])
P_true = cp.array([[1e-7]])
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
x_y_true_all = []
x_y_pred_all = []
x_loss_data = []
x_rmse_loss_data = []
x_rmse_total_data = []
P_y_true_all = []
P_y_pred_all = []
P_loss_data = []
P_rmse_loss_data = []
P_rmse_total_data = []

<<<<<<< HEAD
total_epoch = epoch
=======
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
for epoch in range(epoch + 1):
    x_total_loss = 0
    P_total_loss = 0

    # 創建批次數據
    x_input_data = []
    for i in range(0, traning_size, batch_size):
        batch_x_input_data_all = x_input_data_all[i:i+batch_size] # me
        # 添加到批次列表中
        x_input_data = batch_x_input_data_all# me
        # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
        x_input_tensor = torch.tensor(cp.vstack(x_input_data), dtype=torch.float32).unsqueeze(1).to(device)
        # LSTM進行狀態估計
        x_lstm_output = x_lstm_model(x_input_tensor)

        # 計算損失
        # x_target = torch.tensor(cp.array(x_input_data_all)[1:,:2], dtype=torch.float32).to(device)
        # x_loss = x_loss_fn(x_lstm_output[1:batch_size, :2], x_target[i+1:i+batch_size,:2]) #可以得到一個epoch中每筆資料的mse
        x_target = torch.tensor(cp.array(x_k_update_data)[:, :3], dtype=torch.float32).to(device)
<<<<<<< HEAD

        min1 = cp.min(cp.array(x_k_update_data)[:, 0].get())
        max1 = cp.max(cp.array(x_k_update_data)[:, 0].get())
        norm1 = max1 - min1
        x_lstm_output[:, 0] = (x_lstm_output[:, 0]-min1)/norm1
        x_target[:, 0] = (x_target[:, 0]-min1)/norm1

        min2 = cp.min(cp.array(x_k_update_data)[:, 1].get())
        max2 = cp.max(cp.array(x_k_update_data)[:, 1].get())
        norm2 = max2 - min2
        x_lstm_output[:, 1] = (x_lstm_output[:, 1]-min2)/norm2
        x_target[:, 1] = (x_target[:, 1]-min2)/norm2

        min3 = cp.min(cp.array(x_k_update_data)[:, 2].get())
        max3 = cp.max(cp.array(x_k_update_data)[:, 2].get())
        norm3 = max3 - min3
        x_lstm_output[:, 2] = (x_lstm_output[:, 2]-min3)/norm3
        x_target[:, 2] = (x_target[:, 2]-min3)/norm3

        # x_loss0 = x_loss_fn(x_lstm_output[0:batch_size, 0], x_target[i:i+batch_size, 0])
        # x_loss1 = x_loss_fn(x_lstm_output[0:batch_size, 1], x_target[i:i+batch_size, 1])
        # x_loss2 = x_loss_fn(x_lstm_output[0:batch_size, 2], x_target[i:i+batch_size, 2])
        # print(f'[pos_loss:{x_loss0} -- vel_loss:{x_loss1} -- acc_loss:{x_loss2}]')

        x_loss = x_loss_fn(x_lstm_output[0:batch_size, :3], x_target[i:i+batch_size, :3])
        # x_loss = x_loss_fn(x_lstm_output[0:batch_size, :1], x_target[i:i+batch_size]) #可以得到一個epoch中每筆資料的mse
=======
        x_loss = x_loss_fn(x_lstm_output[0:batch_size, :3], x_target[i:i+batch_size]) #可以得到一個epoch中每筆資料的mse
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
        x_loss_data.append(x_loss.item()) 
        x_rmse_loss = torch.sqrt(x_loss) #可以得到一個epoch中每筆資料的rmse
        x_rmse_loss_data.append(x_rmse_loss.item())
        x_total_loss += x_rmse_loss.item()

        # 保存真實值和預測值
        x_y_true_all.append(x_true.flatten())
        x_y_pred_all.append(x_lstm_output.detach().cpu().numpy().flatten())
        # print("x_y_pred_all =", x_y_pred_all)

        # 反向傳播和參數更新
        x_optimizer.zero_grad()
        x_loss.backward()
        x_optimizer.step()
# ------------------------------------------------------ #
    # --------狀態估測誤差協方差模型-------- #
    P_k_update_data = cp.array(P_k_update_data)
    KCP_data = cp.array(KCP_data)

    # 創建批次數據
    P_input_data = []
    for i in range(0, traning_size, batch_size):
        batch_P_input_data_all = P_input_data_all[i:i+batch_size]# me
        # 添加到批次列表中
        P_input_data = batch_P_input_data_all# me
    
        # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
        P_input_tensor = torch.tensor(cp.vstack(P_input_data), dtype=torch.float32).unsqueeze(1).to(device)

        # LSTM進行狀態估計
        P_lstm_output = P_lstm_model(P_input_tensor)

        # 計算損失
        # P_target = torch.tensor(cp.array(P_input_data_all)[:, :4], dtype=torch.float32).to(device)
        # P_loss = P_loss_fn(P_lstm_output[:, :4], P_target[i:i+batch_size, :4]) #可以得到一個epoch中每筆資料的mse
        P_target = torch.tensor(cp.array(P_input_data_all)[:, :9], dtype=torch.float32).to(device)
        P_loss = P_loss_fn(P_lstm_output[:, :9], P_target[i:i+batch_size, :9]) #可以得到一個epoch中每筆資料的mse
        P_loss_data.append(P_loss.item()) 
        P_rmse_loss = torch.sqrt(P_loss) #可以得到一個epoch中每筆資料的rmse
        P_rmse_loss_data.append(P_rmse_loss.item())
        P_total_loss += P_rmse_loss.item()

        # 保存真實值和預測值
        P_y_true_all.append(P_true.flatten())
        P_y_pred_all.append(P_lstm_output.detach().cpu().numpy().flatten())
        # print("P_y_pred_all =", P_y_pred_all)

        # 反向傳播和參數更新
        P_optimizer.zero_grad()
        P_loss.backward()
        P_optimizer.step()

    x_rmse_total = cp.sqrt(cp.mean(cp.array(x_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
    x_rmse_total_data.append(x_rmse_total)
    P_rmse_total = cp.sqrt(cp.mean(cp.array(P_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
    P_rmse_total_data.append(P_rmse_total)
    if epoch % 1 == 0:
<<<<<<< HEAD
        print(f'-------------------------------------------------------------------')
        print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f} | P_Loss_RMSE : {P_rmse_total.item():.4f}|')
=======
        print(f'---------------------------------------------------------------')
        print(f'[Epoch {epoch} -- x_Loss_RMSE: {x_rmse_total.item():.4f} -- P_Loss_RMSE: {P_rmse_total.item():.4f}]')
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37

# 計算 RMSE
x_y_true_all = cp.array(x_y_true_all)
x_y_pred_all = cp.array(x_y_pred_all)
P_y_true_all = cp.array(P_y_true_all)
P_y_pred_all = cp.array(P_y_pred_all)

os.makedirs('motor/motor_model', exist_ok=True)
# x result儲存模型
<<<<<<< HEAD
torch.save(x_lstm_model.state_dict(), 'motor/motor_model/x_model.pth')
print("-------- x Model saved successfully --------")
# P result儲存模型
torch.save(P_lstm_model.state_dict(), 'motor/motor_model/P_model.pth')
=======
torch.save(x_lstm_model.state_dict(), 'motor/motor_model/x_lstm_kf_model.pth')
print("-------- x Model saved successfully --------")
# P result儲存模型
torch.save(P_lstm_model.state_dict(), 'motor/motor_model/P_lstm_kf_model.pth')
>>>>>>> 306d347394907d950140afa14d4e6ba645070c37
print("-------- P Model saved successfully --------")

# --------狀態估測誤差模型-------- #
plt.figure(figsize=(12, 6))
plt.plot(cp.array(x_rmse_loss_data).get(), label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('x_loss')
plt.legend()
plt.title('x RMSE for every data in each epoch')

plt.figure(figsize=(12, 6))
Epoch = cp.arange(1, len(x_rmse_total_data) + 1)
plt.plot(Epoch.get(), cp.array(x_rmse_total_data).get(), label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('x_loss')
plt.legend()
plt.title('Epoch vs RMSE')

# --------狀態估測誤差協方差模型-------- #
plt.figure(figsize=(12, 6))
plt.plot(cp.array(P_y_pred_all).get()[:, 0], label='X1', color='blue')
plt.plot(cp.array(P_y_pred_all).get()[:, 4], label='X2', color='red')
plt.plot(cp.array(P_y_pred_all).get()[:, 8], label='X2', color='red')
plt.xlabel('Time Step')
plt.ylabel('error cov')
plt.legend()
plt.title('error cov iter')

plt.figure(figsize=(12, 6))
Epoch = cp.arange(1, len(x_rmse_total_data) + 1)
plt.plot(Epoch.get(), cp.array(P_rmse_total_data).get(), label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('P_loss')
plt.legend()
plt.title('Epoch vs RMSE')

plt.show()

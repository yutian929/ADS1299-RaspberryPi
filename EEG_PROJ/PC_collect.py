# 2023/05/25 Zaki: PCSocket(UDL), real-time data reading
# 2023/05/26 Zaki: ues thread "t_rdata" and animation to achieve real-time
# 2023/05/27 Zaki: extend to 8 CHs, add t_wdata and bandpass filter to save data
# 2023/06/08 Zaki: to reduce the latency, the pi only do the spi sampling and UDP transmit to PC
#                  now, the PC get the origin STAT+HC*8 data, need to decoder, preprocess and filter(zjn)
# 2023/07/24 Zaki: due to the failure of ADS1299 equipment, we restart our work now
# 2023/07/31 Zjn: add pred
import os
import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread, Lock
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from preprocessing import preprocessing_single
from NET import CSAC_NET_fff
from Dataset import EEG_Dataset, label2name

'''you can change this to decide whether to read data via UDP'''
EN_rdata = True
'''you can change this to decide whether to write data to excel'''
name = None
EN_wdata = False
file_nums = 0
if EN_wdata:
    name = input('input the head name:')
    if not os.path.exists(name):
        os.makedirs(name)
        print("create folder - ", name)
    else:
        print("folder existed")
    file_path = os.getcwd() + '\\' + name
    files = os.listdir(file_path)
    # Count the number of files
    file_nums = len(files)
    print("existed file nums = ", file_nums)

'''you can change this to decide whether to draw data'''
EN_ddata = False
'''you can change this to decide whether to predict'''
EN_pred = True

# init UDP, PC as Client
serverAddress = ('192.168.137.133', 2222)  # raspberry pi as Server
msgFromClient = "I'm Client"  # validate message
bytesToSend = msgFromClient.encode('utf-8')
bufferSize = 1024
PCSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # init socket
try:
    PCSocket.connect(serverAddress)  # UDP connect
    print("connect successfully")
except:
    print('disconnect')
    exit(0)
PCSocket.sendto(bytesToSend, serverAddress)  # send validate msg
print("already sent validate msg")
print("wait for raspi msg")
data, addr = PCSocket.recvfrom(bufferSize)  # receive from pi
data = data.decode('utf-8')  # data is str
print('Data from Server: ', data)
print('Server IP Address: ', addr[0])  # addr = (IP, port)
print('Server Port: ', addr[1])

# init queue
queues_data = [deque(maxlen=250 * 5) for _ in range(8)]  # to store real-time data -> 8CHs
lock_queues_data = Lock()


# read data from pi via UDP, no data swap
def read_data():
    global queues_data, lock_queues_data, PCSocket, bufferSize, EN_rdata
    time_read_data = 0
    SCALE_TO_UVOLT = 0.0000000121
    while EN_rdata:
        rdata, _ = PCSocket.recvfrom(bufferSize)
        rdata = rdata.decode('utf-8')  # rdata = str([CH1,CH2,...])
        # now, rdata is a str( list( contain  STAT*3 + CHx(1-8)*3 ) )
        rdata = eval(rdata)
        # now, rdata is a list
        if len(rdata) != 27:
            print("len(data) != 27")
            continue
        STAT = rdata[:3]
        CHs = rdata[3:]
        lock_queues_data.acquire()
        for idx in range(8):
            CHx_unpacked = CHs[3 * idx: 3 * idx + 3]
            CHx_packed = struct.pack('3B', CHx_unpacked[0], CHx_unpacked[1], CHx_unpacked[2])
            if CHx_unpacked[0] > 127:
                pre_fix = bytes(bytearray.fromhex('FF'))
            else:
                pre_fix = bytes(bytearray.fromhex('00'))
            CHx_packed = pre_fix + CHx_packed  # now 32 bits, turn into int
            myFloat = struct.unpack('>i', CHx_packed)[0]
            myFloat = myFloat * SCALE_TO_UVOLT + 0.101503
            queues_data[idx].append(myFloat)

        lock_queues_data.release()
        pass
        time_read_data += 1
        if time_read_data % 100 == 0:
            print("time_read_data = ", time_read_data)
    print("EN_rdata = ", EN_rdata)
    return


# write data collected in queues_data to PC bandpass filter, saved as .excel
def write_data():
    global queues_data, lock_queues_data, EN_wdata, name, file_nums
    NUM = 0
    time_save_data = 0
    while EN_wdata:
        if (all(len(channel_data) >= 1250 for channel_data in queues_data)) and (NUM % 500 == 0):
            CHs_data_save = [[] for _ in range(8)]
            lock_queues_data.acquire()
            for i, channel_data in enumerate(queues_data):
                # 提取前1250个数据进行滤波
                # filtered_channel = bandpass_filter(list(channel_data)[:1250], lowcut, highcut, fs)
                CHs_data_save[i] = list(channel_data)[:1250]
                # 剔除前250个数据
                for _ in range(250):
                    channel_data.popleft()
            # 保存滤波后的数据到.xls文件
            channel_names = ['P3', 'Pz', 'P4', 'T5', 'O1', 'OZ', 'O2', 'T6']
            # Create a DataFrame with channel names as column headers
            df = pd.DataFrame({
                channel_names[i]: CHs_data_save[i] for i in range(8)
            })
            file_name = os.path.join(name, name + '_' + str(time_save_data+file_nums) + ".xls")
            df.to_excel(file_name, index=False, header=None)
            time_save_data += 1
            print("time_save_data = ", time_save_data)
            lock_queues_data.release()
        NUM += 1
    # when 'ew' was sent
    print("EN_wdata = ", EN_wdata)
    return


def pred():
    global queues_data, lock_queues_data, EN_pred
    time_wait_pred = 0

    data_pred = torch.zeros(8, 1250)
    pred_dataset = EEG_Dataset('dataset/zyt_fff', 'name2label')
    label_map = pred_dataset.name2label
    model_path = 'wrongmodel/CSAC_fff_23_07_31.pth'
    model = CSAC_NET_fff(ch_in=8, ch_out=512)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    # 调用模型
    pass
    while EN_pred:
        time_wait_pred += 1
        if (all(len(channel_data) >= 1250 for channel_data in queues_data)) and time_wait_pred % 100:
            lock_queues_data.acquire()
            for i, channel_data in enumerate(queues_data):

                data_pred[i] = np.array(list(channel_data)[:1250])

            preprocessed_data = preprocessing_single(data_pred)
            preprocessed_data = np.expand_dims(preprocessed_data, axis=0)
            out = model(preprocessed_data)
            pred_idx = torch.argmax(out, 1).item()
            Similarity = out[0][pred_idx].item()
            pred = label2name(label_map, pred_idx)
            print('Prediction : {} -- Similarity : {}\n'.format(pred, Similarity))
            # queues_data is [deque(CH1[...]), deque(CH2[...]), ...]
            pass
            lock_queues_data.release()
        pass


# 创建图形窗口和子图,初始设置
fig, axs = plt.subplots(8, 1, figsize=(24, 36), sharex=True)
y_min, y_max = -0.1, 0.1
y_ticks = [-0.1, 0, 0.1]
for ax in axs:
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
x_min, x_max = 0, 200  # 假设初始横坐标范围是0到100
axs[-1].set_xlim(x_min, x_max)
lines = []
for ax in axs:
    line, = ax.plot([], [])
    lines.append(line)
data_draw = [[] for _ in range(8)]  # 创建8个空队列


# update animation
def update_plot(frame):
    global queues_data, lock_queues_data, lines, data_draw
    lock_queues_data.acquire()
    for i, queue_data in enumerate(queues_data):
        # print("CH = ", i, 'queue_data = ', queue_data)
        data_draw[i] = list(queue_data)
    # 更新线条数据
    for i, line in enumerate(lines):
        line.set_data(range(len(data_draw[i])), data_draw[i])
    # 更新横坐标范围
    x_max = len(data_draw[0])
    axs[-1].set_xlim(x_max - 200, x_max)  # 只显示最近的200个数据
    lock_queues_data.release()
    return lines


# fork Thread childes
t_rdata = Thread(target=read_data)
t_wdata = Thread(target=write_data)
t_pred = Thread(target=pred)
while True:
    msgFromClient = input('send \'s\' to start streaming\n')
    bytesToSend = msgFromClient.encode('utf-8')
    PCSocket.sendto(bytesToSend, serverAddress)
    if msgFromClient == 'S' or msgFromClient == 's':  # send 's' to sample
        t_rdata.start()
        t_pred.start()
        t_wdata.start()
        if EN_ddata:
            ani = FuncAnimation(fig, update_plot, frames=None, blit=True, interval=500)  # 每200毫秒更新一次图表
            plt.show()
    pass

# 2023/05/29 Zaki: it works! successfully set reg config and triger drdy_callback()
# 2023/05/30 Zaki: add data_process(), conv24bits to 8float, release queue_origin_data_lock earlier
# 2023/05/30 Zaki: add data_send(), another thread to send data to PC via UDP
# 2023/06/07 Zaki: unfortunately, the queue_origin_data_lock caused the severe latency between drdy_callback() and data_process(),
#                  so,i have to cut the data_process(), and send the origin data to PC directly
''' Import '''
from time import sleep
import math
import struct
import socket
from collections import deque
from threading import Thread, Lock
try:
    import spidev
except:
    print("error: import spidev")
    exit(0)
try:
    import RPi.GPIO as GPIO
except:
    print("error: import RPi.GPIO as GPIO")
    exit(0)

''' SPI Init '''
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 400_000
spi.mode = 0b01
spi_lock = Lock()

''' drdy_callback() '''
# time_drdy = 0
num_Bytes_data = 27  # STAT + CHx*8 = 27 Bytes, CHx -> 3*8=24 bits
queue_origin_data = deque()
queue_origin_data_lock = Lock()
time_drdy_callback = 0
ready_UDP = False
time_UDP_send = 0
def drdy_callback(self):
    global CSN, num_Bytes_data, spi, spi_lock, time_drdy_callback
    global RPiSocket, ADDR_CLIENT, ready_UDP, time_UDP_send
    global SCALE_TO_UVOLT
    # 0.data from ads1299 is ready
    # print("time_drdy = ", time_drdy)
    # time_drdy += 1
    # 1.CSn need to be low first
    GPIO.output(CSN, False)
    # 2.read STAT->3Bytes, CHx->3Bytes*8
    spi_lock.acquire()
    data = spi.xfer2([0x00]*num_Bytes_data)
    spi_lock.release()
    # 3.store data into queue_origin_data and let data_process do the next
    # queue_origin_data_lock.acquire()
    # queue_origin_data.append(data)
    # queue_origin_data_lock.release()
    # s
    # print("data = ", data)
    
    # if time_UDP_send <= 3000:
    print('data_raw = ', data)
    ele_origin = data.copy()
    ele_CHs = ele_origin[3:]  # ignore the STAT
    data_final = []  # need to contain 8CHs float data
    for idx in range(8):
        unpacked_CHx = ele_CHs[3*idx : 3*idx + 3]
        packed_CHx = struct.pack('3B', unpacked_CHx[0], unpacked_CHx[1], unpacked_CHx[2])
        # packed_CHx only 24bits, need another 8 bits to form a float
        if (unpacked_CHx[0] > 127):
            pre_fix = bytes(bytearray.fromhex('FF'))
        else:
            pre_fix = bytes(bytearray.fromhex('00'))
        packed_CHx = pre_fix + packed_CHx  # now 32 bits
        myFloat = struct.unpack('>i', packed_CHx)[0]
        myFloat = myFloat*SCALE_TO_UVOLT
        data_final.append(myFloat)
    data_final = [x + 0.101503 for x in data_final]
    print("data_final = ", data_final) 
    
    if ready_UDP:
        # str() and encode it and transfer it to PC via UDP
        # print("data = ", data)
        data = str(data)
        data = data.encode('utf-8')
        RPiSocket.sendto(data, ADDR_CLIENT)
        time_UDP_send += 1
        if time_UDP_send % 100 == 0:
            print('time_UDP_send = ', time_UDP_send)
            
    # 4.CSn need to be High after reading data
    GPIO.output(CSN, True)
#     time_drdy_callback += 1
#     if time_drdy_callback % 100 == 0:
#         print('time_drdy_callback = ', time_drdy_callback)
#     pass

''' data_process() '''

SCALE_TO_UVOLT = 0.0000000121
queue_final_data = deque()
queue_final_data_lock = Lock()
def data_process():
    global SCALE_TO_UVOLT
    global queue_origin_data, queue_origin_data_lock
    global queue_final_data, queue_final_data_lock
    time_data_process = 0
    queue_origin_data.clear()
    while True:
        # 1.acquire queue_origin_data_lock
        queue_origin_data_lock.acquire()
        try:
            # 2.pop the first ele
            ele_origin = queue_origin_data.popleft()
            # 2+.release lock as soon as possible
            queue_origin_data_lock.release()
            time_data_process += 1
            if time_data_process % 100 == 0:
                print("time_data_process = ",time_data_process)
            # 3.conv the ele([STAT_1, STAT_2. STAT_3, CH1_1, CH1_2, CH1_3, CH2_1, ...)
            if len(ele_origin) != 27:
                print("error: len(ele_origin) != 27")
                break
            ele_CHs = ele_origin[3:]  # ignore the STAT
            data_final = []  # need to contain 8CHs float data
            for idx in range(8):
                unpacked_CHx = ele_CHs[3*idx : 3*idx + 3]
                packed_CHx = struct.pack('3B', unpacked_CHx[0], unpacked_CHx[1], unpacked_CHx[2])
                # packed_CHx only 24bits, need another 8 bits to form a float
                if (unpacked_CHx[0] > 127):
                    pre_fix = bytes(bytearray.fromhex('FF'))
                else:
                    pre_fix = bytes(bytearray.fromhex('00'))
                packed_CHx = pre_fix + packed_CHx  # now 32 bits
                myFloat = struct.unpack('>i', packed_CHx)[0]
                myFloat = myFloat*SCALE_TO_UVOLT
                data_final.append(myFloat)
            print("data_final = ", data_final)
            # 4.let data_send do the next
            queue_final_data_lock.acquire()
            queue_final_data.append(data_final)
            queue_final_data_lock.release()
        except:
            queue_origin_data_lock.release()
    pass
# child Thread 1: data_process
t_data_process = Thread(target= data_process)


'''UDP Init'''
SIZE_BUFFER = 1024
PORT_SERVER = 2222
IP_SERVER = '192.168.137.133'  # IP of pi, see it on PC hotpot
RPiSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # init socket
RPiSocket.bind((IP_SERVER, PORT_SERVER))
print('Server is Ready and Listensing...')

''' data_send() '''

def data_send():
    global queue_final_data, queue_final_data_lock, RPiSocket, ADDR_CLIENT
    time_data_send =0
    while True:
        if len(queue_final_data) > 0:
            # 1.acquire queue_final_data_lock
            queue_final_data_lock.acquire()
            # 2.pop the first data final([(float)CH1, CH2,����])
            msg_server = queue_final_data.popleft()
            # print("msg_server = ", msg_server)
            queue_final_data_lock.release()
            # 3.str() and encode it and transfer it to PC via UDP
            msg_server_str = str(msg_server)
            msg_server_encoded = msg_server_str.encode('utf-8')
            # print("msg_server_encode = ", msg_server_encoded)
            RPiSocket.sendto(msg_server_encoded, ADDR_CLIENT)
            time_data_send += 1
            if time_data_send % 100 == 0:
                print("time_data_send = ", time_data_send)
# child Thread 2: data_send
t_data_send = Thread(target= data_send)


''' GPIO Init '''
PWDN = 24  # Board->18  ADS129x->J5-PWDn
CSN = 8  # ->24  ->J3-1 
RESET = 23  # ->16  ->J3-8
START = 22  # ->15  ->J3-14
CLKSEL = 17  # ->11  ->J3-2
DRDY = 25 # ->22  ->J3-15
# MOSI->19  ->J3-11  MISO->21  ->J3-13  SCLK->23  ->J3-3  GND->20  ->J3-4/10/18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWDN, GPIO.OUT)
GPIO.setup(CSN, GPIO.OUT)
GPIO.setup(RESET, GPIO.OUT)
GPIO.setup(START, GPIO.OUT)
GPIO.setup(CLKSEL, GPIO.OUT)
GPIO.setup(DRDY, GPIO.IN)
GPIO.add_event_detect(DRDY, GPIO.FALLING, callback = drdy_callback)

''' Start Up '''
# 1.All Signals LOW
GPIO.output(PWDN, False)
GPIO.output(CLKSEL, False)
GPIO.output(CSN, False)
GPIO.output(RESET, False)
GPIO.output(START, False)
sleep(0.1)
# 2.!CLKSEL HIGH
GPIO.output(CLKSEL, True)
sleep(0.1)
# 3.!PWDN HIGH, !RESET = HIGH
GPIO.output(PWDN, True)
GPIO.output(RESET, True)
sleep(0.1)
# 4.!RESET LOW
GPIO.output(RESET, False)
sleep(0.1)
# 5.!RESET HIGH
GPIO.output(RESET, True)
sleep(0.1)

''' Set ADS1299 Reg Config '''
REG_ID = 0x00  # R->0x3E
REG_CONFIG1 = 0x01  # W->0x96 <Sample Rate>
REG_CONFIG2 = 0x02  # W->0xC0
REG_CONFIG3 = 0x03  # W->0xE0
REG_CH1SET = 0x05  # W*->0x60
REG_BIAS_SENSP = 0x0D  # W->0xFF
REG_BIAS_SENSN = 0x0E  # W->0xFF
RDATAC = 0x10  # cmd, "READ DATA CONTINUES", be used to start sampling
SDATAC = 0x11  # cmd, "STOP DATA CONTINUES", be used to write or read regs
REG_LOFF = 0x0D
REG_GPIO = 0x14
# 0.Ask for lock
spi_lock.acquire()
# 1.Send SDATAC(0x11) then we can RREG and WREG
_ =spi.xfer2([SDATAC]) #xfer sends an array of Bytes and stores the Answer in the specified Variable
print("SDATAC", _)
sleep(0.1)
# 2.Send RREG(0b 001r_rrrr 000n_nnnn) to validate ID ?= 0x3E = 0d62
reg_id = spi.xfer2([0x20|REG_ID, 0, 0])  # [addr, num of regs to set -1, to get feedback val]
print("reg_id = ", reg_id)
if reg_id[2] == 62:
    print("ID validate successfully")
sleep(0.1)
# 3.Send WREG(0b 010r_rrrr 000n_nnnn VAL) to set config regs and RREG
# reg_config1 ?= 0x96 = 0d150, 0x96 -> sample rate = 250, 0x95 -> 500, 0x94 -> 1000, 0x93 -> 2000 
_ =spi.xfer2([0x40|REG_CONFIG1, 0, 0x96])
sleep(0.1)
reg_config1 = spi.xfer2([0x20|REG_CONFIG1, 0, 0])
print("reg_config1 = ", reg_config1)
sleep(0.1)
# reg_config2 ?= 0xC0 = 0d192
_ =spi.xfer2([0x40|REG_CONFIG2, 0, 0xC0])
sleep(0.1)
reg_config2 = spi.xfer2([0x20|REG_CONFIG2, 0, 0])
print("reg_config2 = ", reg_config2)
sleep(0.1)
# reg_config3 ?= 0xE0 = 0d224 
_ =spi.xfer2([0x40|REG_CONFIG3, 0, 0xE0])
sleep(0.1)
reg_config3 = spi.xfer2([0x20|REG_CONFIG3, 0, 0])
print("reg_config3 = ", reg_config3)
sleep(0.1)
# 4.Send WREG to set 8*CHxSET and RREG reg_chx ?= 0x60 = 0d96
for x in range(8):
    _ =spi.xfer2([0x40|(REG_CH1SET + x), 0, 0x60])
    sleep(0.1)
for x in range(8):
    reg_chxset = spi.xfer2([0x20|(REG_CH1SET + x), 0, 0])
    print("reg_ch" + str(x+1) + "set = ", reg_chxset)
    sleep(0.1)
# 5.Send WREG to set BIASSENSP/N and RREG, both should be 0xFF = 0d255
_ =spi.xfer2([0x40|REG_BIAS_SENSP, 0, 0xFF])
sleep(0.1)
reg_biassensp = spi.xfer2([0x20|REG_BIAS_SENSP, 0, 0])
print("reg_biassensp = ", reg_biassensp)
sleep(0.1)
_ =spi.xfer2([0x40|REG_BIAS_SENSN, 0, 0xFF])
sleep(0.1)
reg_biassensn = spi.xfer2([0x20|REG_BIAS_SENSN, 0, 0])
print("reg_biassensn = ", reg_biassensn)
sleep(0.1)
# 6.set LOFF
for l in range(5):
    _ =spi.xfer2([0x40|(REG_LOFF + l), 0, 0x00])
    sleep(0.1)
    reg_loff = spi.xfer2([0x20|REG_LOFF, 0, 0])
    print("reg_loff " + str(l) + "= ", reg_loff)
    sleep(0.1)
# 7.set GPIO
_ =spi.xfer2([0x40|REG_GPIO, 0, 0x00])
sleep(0.1)
reg_gpio = spi.xfer2([0x20|REG_GPIO, 0, 0])
print("reg_gpio = ", reg_gpio)
sleep(0.1)

# 8.see if all right again

print("set sensp/n again")
_ =spi.xfer2([0x40|REG_BIAS_SENSP, 0, 0xFF])
sleep(0.1)
_ =spi.xfer2([0x40|REG_BIAS_SENSN, 0, 0xFF])
sleep(0.1)

reg_id = spi.xfer2([0x20|REG_ID, 0, 0])  # [addr, num of regs to set -1, to get feedback val]
print("reg_id = ", reg_id)
if reg_id[2] == 62:
    print("ID validate successfully")
sleep(0.1)
reg_config1 = spi.xfer2([0x20|REG_CONFIG1, 0, 0])
print("reg_config1 = ", reg_config1)
sleep(0.1)
reg_config2 = spi.xfer2([0x20|REG_CONFIG2, 0, 0])
print("reg_config2 = ", reg_config2)
sleep(0.1)
reg_config3 = spi.xfer2([0x20|REG_CONFIG3, 0, 0])
print("reg_config3 = ", reg_config3)
sleep(0.1)
reg_biassensp = spi.xfer2([0x20|REG_BIAS_SENSP, 0, 0])
print("reg_biassensp = ", reg_biassensp)
sleep(0.1)
reg_biassensn = spi.xfer2([0x20|REG_BIAS_SENSN, 0, 0])
print("reg_biassensn = ", reg_biassensn)
sleep(0.1)

# 5.Send WREG to set BIASSENSP/N and RREG, both should be 0xFF = 0d255
print("set sensp/n again")
_ =spi.xfer2([0x40|REG_BIAS_SENSP, 0, 0xFF])
sleep(0.1)
_ =spi.xfer2([0x40|REG_BIAS_SENSN, 0, 0xFF])
sleep(0.1)



reg_biassensp = spi.xfer2([0x20|REG_BIAS_SENSP, 0, 0])
print("reg_biassensp = ", reg_biassensp)
sleep(0.1)
reg_biassensn = spi.xfer2([0x20|REG_BIAS_SENSN, 0, 0])
print("reg_biassensn = ", reg_biassensn)
sleep(0.1)

# 6.release lock
spi_lock.release()

''' Set Datastream Start '''
spi_lock.acquire()
_ =spi.xfer2([RDATAC])
print("RDATAC", _)
GPIO.output(CSN, True)  # need to be high before drdy_callback
spi_lock.release()
GPIO.output(START, False)

''' It's Show Time in drdy_callback() ! '''
# 1.Wait for Client Validate msg
# msg_client, ADDR_CLIENT = RPiSocket.recvfrom(SIZE_BUFFER)  # validate msg ?= "Hi, I'm Client."
# msg_client_decoded = msg_client.decode('utf-8')
# print('Recvfrom Client: ', msg_client_decoded)
# # 2.Send Validate msg to Server
# msg_server = "Hi, I'm Server."
# msg_server_encoded = msg_server.encode('utf-8')
# RPiSocket.sendto(msg_server_encoded, ADDR_CLIENT)
# 3.Wait for Client msg of Command to Start ADS and GPIO init
# while True:
#     msg_client, _ = RPiSocket.recvfrom(SIZE_BUFFER)
#     msg_client_decoded = msg_client.decode('utf-8')
#     print('From Client: ', msg_client_decoded)
#     if msg_client_decoded == 's' or msg_client_decoded == 'S':  # recv 's' to start SPI
#         ready_UDP = True
#         break
# 4.Start data_process(), data_send()
# t_data_process.start()
# t_data_send.start()
sleep(3)
GPIO.output(START, True)
while True:
    sleep(5)








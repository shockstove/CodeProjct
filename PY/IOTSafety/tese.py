import socket
import binascii
import time

host = "192.168.1.3"
port = 102
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect((host,port))
sock.send(binascii.unhexlify('0300001611e00000001400c1020101c2020101c0010a'))
time.sleep(1)
sock.send(binascii.unhexlify('0300001902f08032010000ccc100080000f0000001000103c0'))
time.sleep(1)
sock.send(binascii.unhexlify('0300002102f0803201000000050010000029000000000009505f50524f4752414d'))
time.sleep(1)
#sock.send(binascii.unhexlify('0300002502f0803201000000330014000028000000000000fd000009505f50524f4752414d'))RUN指令
time.sleep(1)
sock.close()

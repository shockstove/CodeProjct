#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<vector>
#include<fstream>
#include<io.h>
#include<time.h>    
#include<stdlib.h>
#include<iomanip>
#include <Winsock2.h>

#define TIMEOUT_DURATION 2000 // 超时时间，单位：毫秒
#pragma comment(lib, "ws2_32.lib")
using namespace std;

//首部28字节，其余为数据
const int packet_length = 10028;
const int message_length = 10000;

// IP 端口号
string server_IP = "127.0.0.1";
string client_IP = "127.0.0.3";
USHORT server_Port = 8888;
USHORT client_Port = 4444;
int seq;
int ack;
int length = 0;

// 发送、接收缓冲区
char send_buffer[packet_length];
char recv_buffer[packet_length];
char filename[100];
string fimenametmp;
vector<char> dataContent;//数据内容
char timeRecord[20] = { 0 };//时间记录


//加载套接字库
WORD wVersionRequested;//两字节无符号整数。
WSADATA IpWSAData;
SOCKET sockSrv;
//服务器信息
SOCKADDR_IN  Server;
//客户端信息
SOCKADDR_IN Client;

//获取当前时间
void get_time()
{
    time_t t = time(0);
    strftime(timeRecord, sizeof(timeRecord), "%H:%M:%S", localtime(&t));
}

//设置seq和ack
void set_seq_Ack(int newSeq, bool newAck)
{
    seq = newSeq;
    //从接收缓冲区recv_buffer中获取4个字节并将其拼接成一个整数值，用于获取接收到的序列号
    int recv_seq = (int)((unsigned char)recv_buffer[12] << 24) + (int)((unsigned char)recv_buffer[13] << 16) + (int)((unsigned char)recv_buffer[14] << 8) + (int)(unsigned char)recv_buffer[15];
    //如果需要发送确认应答，ack=recv_seq+1，否则ack=0
    ack = newAck ? recv_seq + 1 : 0;
    //Seq
    send_buffer[12] = (char)(seq >> 24);//高8
    send_buffer[13] = (char)(seq >> 16);//次高8
    send_buffer[14] = (char)(seq >> 8);//次低8
    send_buffer[15] = (char)seq;//最低8
    //Ack
    send_buffer[16] = (char)(ack >> 24);
    send_buffer[17] = (char)(ack >> 16);
    send_buffer[18] = (char)(ack >> 8);
    send_buffer[19] = (char)ack;
}

void set_length(int len)
{
    length = len;//设置数据长度
    send_buffer[20] = (char)(length >> 24);
    send_buffer[21] = (char)(length >> 16);
    send_buffer[22] = (char)(length >> 8);
    send_buffer[23] = (char)length;
}

//清除标志位
void clear_flag()
{
    send_buffer[24] = 0;
    send_buffer[25] = 0;
}

//确认ACK，1表明确认号有效
void set_ACK()
{
    send_buffer[24] += 0xF0;
}

//同步SYN
//SYN = 1、ACK = 0，该报文段是连接请求报文段
//SYN = 1、ACK = 1，该报文段是同意连接报文段
void set_SYN()
{
    send_buffer[24] += 0xF;
}

//终止FIN
//FIN = 1，该报文段请求释放连接
void set_FIN()
{
    send_buffer[25] += 0xF0;
}

//复位RST
//RST = 1，连接需要释放并重新建立
void set_ST()
{
    send_buffer[25] += 0xC;
}

//结束OV，为1表示接收完成
void set_OV()
{
    send_buffer[25] += 0x3;
}

//各标志位对应检查
bool check_ACK()
{
    return recv_buffer[24] & 0xF0;
}
bool check_SYN()
{
    return recv_buffer[24] & 0xF;
}
bool check_FIN()
{
    return recv_buffer[25] & 0xF0;
}
bool check_ST()
{
    return recv_buffer[25] & 0xC;
}
bool check_OV()
{
    return recv_buffer[25] & 0x3;
}

bool check_seq() //检查seq，seq是计算出来的
{
    //将接收缓冲区中连续的4个字节依次转换为整数表示的序列号recv_seq
    int recv_seq = (int)((unsigned char)recv_buffer[12] << 24) + (int)((unsigned char)recv_buffer[13] << 16) + (int)((unsigned char)recv_buffer[14] << 8) + (int)(unsigned char)recv_buffer[15];
    //判断recvSeq是否等于期望的确认号ack，或者等于0
    if ((recv_seq == ack) || (recv_seq == 0))
        return true;
    else
    {
        cout << "错误，可能出现丢包或重复包" << endl;
        cout << recv_seq << " " << ack << endl;
        return false;
    }
}

//检查ack值
bool check_ack_value()
{
    //// 从接收缓冲区中提取四个字节并将它们拼接成32位整数recvack
    int recv_ack = (int)((unsigned char)recv_buffer[16] << 24) + (int)((unsigned char)recv_buffer[17] << 16) + (int)((unsigned char)recv_buffer[18] << 8) + (int)(unsigned char)recv_buffer[19];
    // 检查recvack是否等于期望的确认号seq加1
    if (recv_ack == seq + 1)
        return true;
    else
    {
        cout << "错误，可能出现丢包或重复包" << endl;
        // 输出发送缓冲区和接收缓冲区中相关字节的值
        cout << (int)send_buffer[12] << " " << (int)send_buffer[13] << " " << (int)send_buffer[14] << " " << (int)send_buffer[15] << endl;
        cout << (unsigned int)recv_buffer[16] << " " << (unsigned int)recv_buffer[17] << " " << (unsigned int)recv_buffer[18] << " " << (int)recv_buffer[19] << endl;
        cout << recv_ack << endl;
        return false;
    }
}

//计算校验和
void set_Checksum()
{
    int sum = 0;
    //计算前26字节，即除了校验和的头部
    for (int i = 0; i < packet_length; i += 2)
    {
        if (i == 26)
            continue;
        //在每次循环中，将两个字节拼接成16位，并加到sum中
        sum += (send_buffer[i] << 8) + send_buffer[i + 1];
        //超过了16位的最大值0x10000
        //出现进位,将进位部分移至最低位
        if (sum >= 0x10000)
        {
            sum -= 0x10000;
            sum += 1;
        }
    }

    //对sum取反
    USHORT checkSum = ~(USHORT)sum;
    //高8位和低8位存到对应位置
    send_buffer[26] = (char)(checkSum >> 8);
    send_buffer[27] = (char)checkSum;
}

//校验码检查
bool check_Checksum()
{
    int sum = 0;
    //计算前26字节，即除了校验和的头部
    for (int i = 0; i < packet_length; i += 2)
    {
        if (i == 26)
            continue;
        //在每次循环中，将两个字节拼接成16位，并加到sum中
        sum += (recv_buffer[i] << 8) + recv_buffer[i + 1];
        //超过了16位的最大值0x10000
        //出现进位,将进位部分移至最低位
        if (sum >= 0x10000)
        {
            sum -= 0x10000;
            sum += 1;
        }
    }

    //判断校验和和sum的和是否等于0xffff
    USHORT checksum = (recv_buffer[26] << 8) + (unsigned char)recv_buffer[27];
    if (checksum + (USHORT)sum == 0xffff)
    {
        return true;
    }
    else
    {
        cout << "校验码校验失败" << endl;
        return false;
    }
}

//发送日志
void sendlog()
{
    get_time();
    cout << timeRecord << " [send] ";
    int seqtmp = (int)((unsigned char)send_buffer[12] << 24) + (int)((unsigned char)send_buffer[13] << 16) + (int)((unsigned char)send_buffer[14] << 8) + (int)(unsigned char)send_buffer[15];
    int acktmp = (int)((unsigned char)send_buffer[16] << 24) + (int)((unsigned char)send_buffer[17] << 16) + (int)((unsigned char)send_buffer[18] << 8) + (int)(unsigned char)send_buffer[19];
    int lengthtmp = (int)((unsigned char)send_buffer[20] << 24) + (int)((unsigned char)send_buffer[21] << 16) + (int)((unsigned char)send_buffer[22] << 8) + (int)(unsigned char)send_buffer[23];
    cout << "  Seq: " << setw(5) << setiosflags(ios::left) << seqtmp << "Ack: " << setw(5) << setiosflags(ios::left) << acktmp << "Length: " << setw(5) << setiosflags(ios::left) << lengthtmp;
    int ACKtmp = (send_buffer[24] & 0xF0) ? 1 : 0;
    int syntmp = (send_buffer[24] & 0xF) ? 1 : 0;
    int fintmp = (send_buffer[25] & 0xF0) ? 1 : 0;
    int sttmp = (send_buffer[25] & 0xC) ? 1 : 0;
    int ovtmp = (send_buffer[25] & 0x3) ? 1 : 0;
    cout << "  ACK: " << ACKtmp << " SYN: " << syntmp << " FIN: " << fintmp << " ST: " << sttmp << " OV: " << ovtmp << " Checksum: " << ((unsigned char)send_buffer[26] << 8) + (unsigned char)send_buffer[27] << endl;
}

// 接收日志
void recvlog()
{
    get_time();
    cout << timeRecord << " [recv] ";
    int seqtmp = (int)((unsigned char)recv_buffer[12] << 24) + (int)((unsigned char)recv_buffer[13] << 16) + (int)((unsigned char)recv_buffer[14] << 8) + (int)(unsigned char)recv_buffer[15];
    int acktmp = (int)((unsigned char)recv_buffer[16] << 24) + (int)((unsigned char)recv_buffer[17] << 16) + (int)((unsigned char)recv_buffer[18] << 8) + (int)(unsigned char)recv_buffer[19];
    int lengthtmp = (int)((unsigned char)recv_buffer[20] << 24) + (int)((unsigned char)recv_buffer[21] << 16) + (int)((unsigned char)recv_buffer[22] << 8) + (int)(unsigned char)recv_buffer[23];
    cout << "  Seq: " << setw(5) << setiosflags(ios::left) << seqtmp << "Ack: " << setw(5) << setiosflags(ios::left) << acktmp << "Length: " << setw(5) << setiosflags(ios::left) << lengthtmp;
    int ACKtmp = (recv_buffer[24] & 0xF0) ? 1 : 0;
    int syntmp = (recv_buffer[24] & 0xF) ? 1 : 0;
    int fintmp = (recv_buffer[25] & 0xF0) ? 1 : 0;
    int sttmp = (recv_buffer[25] & 0xC) ? 1 : 0;
    int ovtmp = (recv_buffer[25] & 0x3) ? 1 : 0;
    cout << "  ACK: " << ACKtmp << " SYN: " << syntmp << " FIN: " << fintmp << " ST: " << sttmp << " OV: " << ovtmp << " Checksum: " << ((unsigned char)send_buffer[26] << 8) + (unsigned char)send_buffer[27] << endl;
}


//回复连接请求
void connect_Ack_prepare()
{
    //设置源端口与目的端口号，因为始终保持连接，所以不需要再更改。
    send_buffer[0] = (char)(server_Port >> 8);
    send_buffer[1] = (char)(server_Port & 0xFF);
    send_buffer[2] = (char)(client_Port >> 8);
    send_buffer[3] = (char)(client_Port & 0xFF);
    //设置源IP与目标IP
    int tmp = 0;
    int ctrl = 4;
    //解析服务器IP地址
    for (int i = 0; i < server_IP.length(); i++)
    {
        if (server_IP[i] == '.')
        {
            send_buffer[ctrl++] = (char)tmp;
            tmp = 0;
        }
        else
        {
            tmp += tmp * 10 + (int)server_IP[i] - 48;
        }
    }
    send_buffer[ctrl++] = (char)tmp;
    tmp = 0;

    //解析客户端ip地址
    for (int i = 0; i < client_IP.length(); i++)
    {
        if (client_IP[i] == '.')
        {
            send_buffer[ctrl++] = (char)tmp;
            tmp = 0;
        }
        else
        {
            tmp += tmp * 10 + (int)client_IP[i] - 48;
        }
    }
    send_buffer[ctrl++] = (char)tmp;

    //初始序号与确认序号
    //确认序号为收到的初始序号加一
    set_seq_Ack(rand(), 1);

    //握手报文长度设置为0，因为这是第一个连接确认报文
    set_length(0);

    //设置标志位，第二次握手需要设置ACK和SYN
    clear_flag();
    set_ACK();
    set_SYN();

    //报文数据部分置0
    for (int i = 28; i < packet_length; i++)
        send_buffer[i] = 0;

    //计算并设置校验和
    set_Checksum();
}


void initial()
{
    // 指定使用的Winsock版本1.1
    wVersionRequested = MAKEWORD(1, 1);
    //进行socket库的绑定
    int error = WSAStartup(wVersionRequested, &IpWSAData);
    if (error != 0)
    {
        cout << "初始化错误" << endl;
        exit(0);
    }

    //创建用于监听的socket
    // AF_INET表示使用TCP / IP和IPv4协议，SOCK_DGRAM表示这是一个UDP数据报的socket
    sockSrv = socket(AF_INET, SOCK_DGRAM, 0);


    //IP地址
    Server.sin_addr.s_addr = inet_addr(server_IP.c_str());
    //协议簇
    Server.sin_family = AF_INET;
    //连接端口号
    Server.sin_port = htons(server_Port);

    //绑定端口
    bind(sockSrv, (SOCKADDR*)&Server, sizeof(SOCKADDR));//会返回一个SOCKET_ERROR
    unsigned long ul = 1;
    int ret = ioctlsocket(sockSrv, FIONBIO, (unsigned long*)&ul);
}

//数据传输
void transmission(int ctrl)
{
    int len = sizeof(SOCKADDR);
    int start;//记录时间
    bool whether;//是否接收到数据
    int retransmissionCount = 0;//记录重传次数
    char lastSendBuffer[packet_length];  // 保存上一次发送的数据

    switch (ctrl)
    {
    case 0:
        //只发发送不接受
        sendto(sockSrv, send_buffer, sizeof(send_buffer), 0, (SOCKADDR*)&Client, len);
        sendlog();
        break;

    case 1:
        //只接收不发送数据
        while (1)//循环接收并打印日志
        {
            if (recvfrom(sockSrv, recv_buffer, sizeof(recv_buffer), 0, (SOCKADDR*)&Client, &len) > 0)
            {
                recvlog();
                break;
            }
        }
        break;
    case 2:
        //先发送再接收数据
        do
        {
            //memcpy(lastSendBuffer, send_buffer, sizeof(send_buffer));//复制上一条数据

            sendto(sockSrv, send_buffer, sizeof(send_buffer), 0, (SOCKADDR*)&Client, len);
            sendlog();
            start = clock();
            whether = false;
            while (clock() - start < TIMEOUT_DURATION) //在规定时间内接收数据
            {
                if (recvfrom(sockSrv, recv_buffer, sizeof(recv_buffer), 0, (SOCKADDR*)&Client, &len) > 0)
                {
                    whether = true;
                    recvlog();
                    break;
                }
            }
            if (!whether)//规定时间内没有获得，说明超时
            {
                // 超时处理
                // 记录超时次数，并重新发送数据，并打印日志
                retransmissionCount++;
                cout << "超时，重传数据，重传次数：" << retransmissionCount << endl;
                sendto(sockSrv, lastSendBuffer, sizeof(lastSendBuffer), 0, (SOCKADDR*)&Server, len);
                sendlog();
            }
        } while (!check_Checksum() || (!whether));

        break;
    default:
        break;
    }
}


int main()
{
    initial();
    cout << "—————— 服务器运行成功 ——————" << endl;
    cout << "—————— 正在等待连接 ——————" << endl;
    while (1)
    {
        transmission(1);
        //等待接收连接申请
        if (check_Checksum() && check_SYN())
        {
            //回复ack
            connect_Ack_prepare();
            while (1)
            {
                transmission(2);
                cout << "—————— 握手成功，连接结束 ——————" << endl;
                if (check_ACK() && check_ack_value())
                {
                    //收到的三次握手正确
                    break;
                }
            }
            break;
        }
        else {
            //收到的一次握手错误，回复ACK=0，让其重传。回到while循环继续等待接收。
            connect_Ack_prepare();
            clear_flag();
            set_Checksum();
            transmission(0);
        }
    }
    ack = 0;
    //先接后发
    while (1)
    {
        transmission(1);

        if (check_Checksum())
        {
            if (check_FIN())
            {
                break;
            }
            if (check_ST())
            {
                //ST信号，收到的是文件名，一个这一批次的第一个报文。
                //这是第一个文件
                int messageLength = (int)(recv_buffer[20] << 24) + (int)(recv_buffer[21] << 16) + (int)(recv_buffer[22] << 8) + (int)(recv_buffer[23]);
                for (int i = 0; i < messageLength; i++)
                {
                    filename[i] = recv_buffer[28 + i];
                }
                fimenametmp = "C:\\Users\\lhw\\Desktop\\Network\\" + (string)filename;
                cout << "开始接收文件：" << fimenametmp << endl;
            }
            if (check_OV())
            {
                //OV信号，文件结束
                int messageLength = (int)(recv_buffer[20] << 24) + (int)(recv_buffer[21] << 16) + (int)(recv_buffer[22] << 8) + (int)(recv_buffer[23]);
                for (int i = 0; i < messageLength; i++)
                {
                    dataContent.push_back(recv_buffer[28 + i]);
                }
                ofstream fout(fimenametmp.c_str(), ofstream::binary);
                for (int i = 0; i < dataContent.size(); i++)
                {
                    fout << dataContent[i];
                }
                vector<char>().swap(dataContent);

                cout << "文件：" << fimenametmp << "接收成功！" << endl;
            }
            if ((recv_buffer[25] == 0) && check_seq())
            {
                //正常的报文
                int messageLength = (int)(recv_buffer[20] << 24) + (int)(recv_buffer[21] << 16) + (int)(recv_buffer[22] << 8) + (int)(recv_buffer[23]);
                for (int i = 0; i < messageLength; i++)
                {
                    dataContent.push_back(recv_buffer[28 + i]);
                }
            }
            set_seq_Ack(0, 1);
            set_length(0);
            clear_flag();
            set_ACK();
        }
        else
        {
            clear_flag();
        }
        set_Checksum();
        transmission(0);
    }
    cout << "—————— 断连开始 ——————" << endl;
    //收到了第一次挥手
    set_seq_Ack(rand(), 1);
    set_length(0);
    clear_flag();
    set_FIN();
    set_ACK();
    set_Checksum();
    transmission(0);

    set_seq_Ack(rand(), 1);
    set_length(0);
    clear_flag();
    set_ACK();
    set_FIN();
    set_Checksum();
    do
    {
        transmission(2);
    } while (!(check_Checksum() && check_ACK() && check_ack_value()));//发收各一次



    closesocket(sockSrv);
    WSACleanup();
    cout << "—————— 挥手成功，断连结束 ——————" << endl;
}




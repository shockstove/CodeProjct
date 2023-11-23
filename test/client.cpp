#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 
#include<iostream>
#include<vector>
#include<fstream>
#include<time.h> 
#include<io.h>
#include<string>
#include<iomanip>
#include <Winsock2.h>

#define TIMEOUT_DURATION 2000 // 超时时间，单位：毫秒
#pragma comment(lib, "ws2_32.lib")
using namespace std;

//首部28字节，其余为数据
const int packet_length = 10028;
const int message_length = 10000;

// IP 端口号
string server_IP = "127.0.0.3";
string client_IP = "127.0.0.2";
USHORT server_Port = 4444;
USHORT client_Port = 1111;
int length = 0;
int seq;
int ack;
vector<string> files;
string path("E:\\W\\test\\helloworld.txt");

// 发送、接收缓冲区
char send_buffer[packet_length];
char recv_buffer[packet_length];
vector<char> data_content; // 数据内容
char time_record[20] = { 0 }; //时间记录


//加载套接字库
WORD wVersionRequested; //两字节无符号整数
WSADATA IpWSAData;
SOCKET sockSrv;
//要连接的服务器
SOCKADDR_IN  Server;
//客户端信息
SOCKADDR_IN Client;

// 获取当前时间
void get_time()
{
    time_t t = time(0);
    strftime(time_record, sizeof(time_record), "%H:%M:%S", localtime(&t));
}

//设置seq和ack
void set_seq_Ack(int newSeq, bool newAck)
{
    seq = newSeq;
    //从接收缓冲区recv_buffer中获取4个字节并将其拼接成一个整数值，用于获取接收到的序列号
    int recv_seq = (int)((unsigned char)recv_buffer[12] << 24) + (int)((unsigned char)recv_buffer[13] << 16) + (int)((unsigned char)recv_buffer[14] << 8) + (int)(unsigned char)recv_buffer[15];
    //如果需要发送确认应答，ack=recvSeq+1，否则ack=0
    ack = newAck ? recv_seq + 1 : 0;
    //Seq
    send_buffer[12] = (char)(seq >> 24);
    send_buffer[13] = (char)(seq >> 16);
    send_buffer[14] = (char)(seq >> 8);
    send_buffer[15] = (char)seq;
    //Ack
    send_buffer[16] = (char)(ack >> 24);
    send_buffer[17] = (char)(ack >> 16);
    send_buffer[18] = (char)(ack >> 8);
    send_buffer[19] = (char)ack;
}

//设置数据长度
void set_length(int len)
{
    length = len;
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

//检查ack值
bool check_ack_value()
{
    int recv_ack = (int)((unsigned char)recv_buffer[16] << 24) + (int)((unsigned char)recv_buffer[17] << 16) + (int)((unsigned char)recv_buffer[18] << 8) + (int)(unsigned char)recv_buffer[19];
    if (recv_ack == seq + 1)
        return true;
    else
    {
        cout << "错误，可能出现丢包或重复包" << endl;
        cout << (int)send_buffer[12] << " " << (int)send_buffer[13] << " " << (int)send_buffer[14] << " " << (int)send_buffer[15] << endl;
        cout << (unsigned int)recv_buffer[16] << " " << (unsigned int)recv_buffer[17] << " " << (unsigned int)recv_buffer[18] << " " << (int)recv_buffer[19] << endl;
        cout << recv_ack << endl;
        return false;
    }
}

//计算校验和
void set_CheckSum()
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
    // 计算前26字节，即除了校验和的头部
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
    {
        cout << "校验码校验失败" << endl;
        return false;
    }
}

//发送日志
void sendlog()
{
    get_time();
    cout << time_record << " [send] ";
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
    cout << time_record << " [recv] ";
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



//建立连接
void connect_establishment()
{
    //设置源端口与目的端口号，因为始终保持连接，所以不需要再更改。
    send_buffer[0] = (char)(client_Port >> 8);
    send_buffer[1] = (char)(client_Port & 0xFF);
    send_buffer[2] = (char)(server_Port >> 8);
    send_buffer[3] = (char)(server_Port & 0xFF);
    //设置源IP与目标IP
    int tmp = 0;
    int ctrl = 4;
    //解析客户端IP地址
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
    tmp = 0;

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

    //初始序号与确认序号
    set_seq_Ack(rand(), 0);

    //握手报文长度为0，第一个连接请求报文
    set_length(0);
    //清除标志位设置SYN
    clear_flag();
    set_SYN();
    //数据部分设0
    for (int i = 28; i < packet_length; i++)
        send_buffer[i] = 0;
    //计算并设置校验和
    set_CheckSum();
}



// 获取文件列表
bool getFiles(string path, vector<string>& files)
{
    //文件句柄
    u_int64 hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("").c_str(), &fileinfo)) != -1) {
        if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
        {
            // 保存文件的全路径
            files.push_back(p.assign(path).append("\\").append(fileinfo.name));

        }
        while (_findnext(hFile, &fileinfo) == 0)
        {
            if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
            {
                // 保存文件的全路径
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));

            }
        }
        _findclose(hFile);
    }
    return 1;
}

//加载文件到内存
void loadFile(int number, long long& fileSize)
{
    ifstream fin(files[number].c_str(), ios::binary | ios::ate);
    fileSize = fin.tellg();
    fin.seekg(0, ios::beg);

    vector<char>().swap(data_content);
    char t = fin.get();
    while (fin)
    {
        data_content.push_back(t);
        t = fin.get();
    }
}

// 初始化
void initial()
{
    //MAKEWORD，第一个参数为低位字节，主版本号，第二个参数为高位字节，副版本号
    wVersionRequested = MAKEWORD(1, 1);
    //进行socket库的绑定
    int error = WSAStartup(wVersionRequested, &IpWSAData);
    if (error != 0)
    {
        cout << "初始化错误" << endl;
        exit(0);
    }

    //创建用于监听的socket
    //AF_INET：TCP/IP&IPv4，SOCK_DGRAM：UDP数据报
    sockSrv = socket(AF_INET, SOCK_DGRAM, 0);
    //IP地址
    Server.sin_addr.s_addr = inet_addr(server_IP.c_str());
    //协议簇
    Server.sin_family = AF_INET;
    //连接端口号
    Server.sin_port = htons(server_Port);
    unsigned long ul = 1;
    int ret = ioctlsocket(sockSrv, FIONBIO, (unsigned long*)&ul);
}

//数据传输
void transmission(int ctrl)
{
    int len = sizeof(SOCKADDR);
    int start;//记录时间
    bool whether;//是否接收到数据
    int retransmissionCount = 0;
    char lastSendBuffer[packet_length];  // 保存上一次发送的数据

    switch (ctrl)
    {

    case 0:
        whether = false;
        do
        {
            sendto(sockSrv, send_buffer, sizeof(send_buffer), 0, (SOCKADDR*)&Server, len);
            sendlog();
            whether = false;
            start = clock();
            while (clock() - start < TIMEOUT_DURATION)//不超时
            {
                if (recvfrom(sockSrv, recv_buffer, sizeof(recv_buffer), 0, (SOCKADDR*)&Server, &len) > 0)
                {
                    recvlog();
                    whether = true;
                    break;
                }
            }
            //if (!whether)//规定时间内没有获得，说明
            //{
            //    retransmissionCount++;
            //    cout << "超时，重传数据，重传次数：" << retransmissionCount << endl;
            //}
        } while (whether);

        break;
    case 1:
        while (1)
        {
            if (recvfrom(sockSrv, recv_buffer, sizeof(recv_buffer), 0, (SOCKADDR*)&Server, &len) > 0)
            {
                recvlog();
                break;
            }
        }
        break;

    case 2:
        //先发再收
        do
        {
            sendto(sockSrv, send_buffer, sizeof(send_buffer), 0, (SOCKADDR*)&Server, len);
            sendlog();
            start = clock();
            whether = false;
            while (clock() - start < TIMEOUT_DURATION)
            {
                if (recvfrom(sockSrv, recv_buffer, sizeof(recv_buffer), 0, (SOCKADDR*)&Server, &len) > 0)
                {
                    recvlog();
                    whether = true;//规定时间内获得了
                    break;
                }
            }

            if (!whether)//规定时间内没有获得，说明
            {
                retransmissionCount++;
                cout << "超时，重传数据，重传次数：" << retransmissionCount << endl;
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
    cout << "—————— 客户端运行成功 ——————" << endl;
    cout << "—————— 连接开始 ——————" << endl;
    connect_establishment();
    while (1)
    {
        //第一次握手，校验失败会重传。
        transmission(2);
        if (check_ACK() && check_SYN() && check_ack_value())
        {
            //收到的包正确且为二次握手
            //准备进行三次握手的回复
            set_seq_Ack(rand(), 1);
            //标志位
            clear_flag();
            set_ACK();
            set_CheckSum();
            transmission(0);
            cout << "—————— 握手成功，连接结束 ——————" << endl;
            break;

        }
    }
    getFiles(path, files);
    int num = files.size();
    cout << "—————— 开始发送文件 ——————" << endl;
    clock_t begin_time = clock();
    clock_t end_time = clock();
    for (int i = 0; i < num; i++)
    {
        begin_time = clock();
        char fileName[100];
        strcpy(fileName, files[i].substr(20, files[i].length() - 20).c_str());
        cout << "开始发送文件：" << fileName << endl;
        //要发送的第一条消息就是文件的名字
        set_seq_Ack(0, 0);
        set_length(strlen(fileName));
        clear_flag();
        set_ST();
        //报文
        for (int j = 0; j < length; j++)
        {
            send_buffer[j + 28] = fileName[j];
        }
        set_CheckSum();
        do
        {
            transmission(2);
        } while (!(check_ACK() && check_ack_value()));

        //将文件加载到dataContent动态数组中。
        long long fileSize;
        loadFile(i, fileSize);
        for (int j = 0; j < data_content.size(); j++)
        {
            //装载数据报文
            send_buffer[28 + (j % message_length)] = data_content[j];
            if ((j % message_length == message_length - 1) && (j != data_content.size() - 1))
            {
                //填满了一组报文，准备发送
                set_seq_Ack(seq + 1, 0);
                set_length(message_length);
                clear_flag();
                set_CheckSum();
                do
                {
                    transmission(2);
                } while (!(check_ACK() && check_ack_value()));
            }
            if (j == data_content.size() - 1)
            {
                //最后一组报文
                set_seq_Ack(seq + 1, 0);
                set_length(j % message_length + 1);
                clear_flag();
                set_OV();
                set_CheckSum();
                do
                {
                    transmission(2);
                } while (!(check_ACK() && check_ack_value()));
            }
        }
        end_time = clock();
        cout << "文件：" << fileName << " 发送成功！" << endl;
        cout << "文件大小：" << fileSize << " Bytes" << endl;
        cout << "传输用时: " << (end_time - begin_time) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
        cout << "吞吐率：" << fileSize * 8.0 / ((end_time - begin_time) / CLOCKS_PER_SEC) << "bps" << endl;
    }
    cout << "—————— 文件发送完成 ——————" << endl;
    //挥手
    cout << "—————— 断连开始 ——————" << endl;
    set_seq_Ack(rand(), 0);
    set_length(0);
    clear_flag();
    set_FIN();
    set_CheckSum();
    do
    {
        transmission(2);
    } while (!(check_Checksum() && check_ACK() && check_ack_value()));//发收各一次

    while (1)
    {
        transmission(1);

        if (check_Checksum())
        {
            if (check_FIN())
            {
                break;
            }
        }
    }

    set_seq_Ack(rand(), 1);
    set_length(0);
    clear_flag();
    set_FIN();
    set_ACK();
    set_CheckSum();
    transmission(0);

    closesocket(sockSrv);
    WSACleanup();
    cout << "—————— 挥手成功，断连结束 ——————" << endl;

}




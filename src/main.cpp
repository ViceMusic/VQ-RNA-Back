#include <iostream>
#include "test.h"
#include "operatehttp.h"
#include <cstring>     // for memset
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>    // for close
#include <arpa/inet.h> // for inet_ntoa
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <string>
#include <map>
using namespace std;

//爆出错误
void perror(string err){
    std::cerr<<err<<std::endl;
}

//根据端口创建套接字
int create_server_socket(int* port){
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Error creating socket.");
        return -1;
    }
    return server_fd;
}

//客户套接字接收客户要求并且进行处理
void operate_in_client_server(int* client_fd,std::string ip){
    //字符数组作为接收工具，但是字符数组后面并没有字符串截至符号所以不会被视为截至。
    char buffer[1024];
    std::string http_request;
    ssize_t bytes_received;
    //循环读取每一行的数据
    while ((bytes_received = recv(*client_fd, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_received] = '\0'; // 添加字符串结束符
        http_request += buffer; // 拼接接收到的数据
        
        // 检查请求是否完整
        if (http_request.find("\r\n\r\n") != std::string::npos) {
            break; // 找到请求结束的标志
        }
    }
    std::cout<<"接收到的请求内容为"<<http_request<<std::endl;
    //判断出现问题的地方， 如果没有问题就打印http的具体信息
    if (bytes_received < 0) {
        std::cerr << "Error receiving data." << std::endl;
    } else {
        std::cout << "Received HTTP Request:" << std::endl;
        //调用请求处理方法
        std::map<std::string,std::string> req= operate_http_request(http_request);
        //根据请求信息进行日志记录（就不记录响应状态了）

        //处理请求
        operate_http_response(req,client_fd,ip);
    }

    //关闭套接字
    close(*client_fd);
    //使用文件操作符发送信息
    // const char *message = "已经收到消息， 一切都好";
    // ssize_t bytes_sent = send(*client_fd, message, strlen(message), 0);
    // if (bytes_sent < 0) {
    //     std::cerr << "Error sending message." << std::endl;
    // } else {
    //     std::cout << "Sent " << bytes_sent << " bytes." << std::endl;
    // }


}

//线程的阻塞


//线程池类， 是用的是消费者，生产者模型（有点反直觉， 因为资源是任务， 所以消费者是多个线程）
class ThreadsPool{
public:
    //线程数组
    std::vector<std::thread> threads;
    //任务队列
    std::queue<std::function<void()>> tasks;
    //标志
    bool stop;
    //互斥锁
    std::mutex mtx;
    //条件变量
    std::condition_variable cv;
    //构造函数
    ThreadsPool(int num):stop(false){
        for(int i=0;i<num;i++){
            //emplace_back可以理解为直接调用构造函数
            threads.emplace_back([this,i]{
                //直接使用lambad表达式创建线程
                while(1){
                    //创建锁
                    std::unique_lock<std::mutex> lock(this->mtx);
                    //启动条件变量, 如果程序已经停止执行或者任务队列不为空，则可以继续执行，否则就进行等待
                    this->cv.wait(lock,[this]{return stop||!tasks.empty();});

                    //如果是程序停止了， 就消除这个线程
                    if(stop){
                        return;
                    }

                    //从队列中取出任务， 并且进行执行，
                    std::function<void()> task(std::move(tasks.front()));
                    //将这个任务弹出stack
                    tasks.pop();
                    //已经取出， 取消锁定， 并且执行任务
                    lock.unlock();
                    //执行对应任务， 直接调用
                    task();
                    cout<<"线程"<<i<<"进行服务"<<endl;
                }
            });
        }
    }
    //系出构造函数
    ~ThreadsPool(){
        //强制执行完成所有的任务
        {
            std::unique_lock<std::mutex> lock(this->mtx);
            this->stop=true;
        }
        //通知所有线程启动
        cv.notify_all();
        //等待所有线程结束
        for(auto& t : threads){
            t.join();
        }

    }
    //向线程池中加入任务
    template<typename Func, typename... Args>
    void enqueue(Func func,Args... args){
        //放入一个任务
        {
            std::unique_lock<std::mutex> lock(this->mtx);
            tasks.emplace([=](){func(args...);});
        }
        //通知条件变量来执行任务
        cv.notify_one();
    }


};


int main(){
    int port=8080;
    // 1. 获取套接字， 根据套接字创建客户套接字
    int server_fd=create_server_socket(&port);

    // 2. 设置套接字选项
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options." << std::endl;
        close(server_fd);
        return -1;
    }

    // 3. 定义服务器地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr)); // 清零
    server_addr.sin_family = AF_INET;             // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY;    // 监听所有可用接口
    server_addr.sin_port = htons(8080);           // 监听端口 8080

    // 4. 绑定套接字
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error binding socket." << std::endl;
        close(server_fd);
        return -1;
    }

    // 5. 开始监听
    if (listen(server_fd, 5) < 0) {
        std::cerr << "Error listening on socket." << std::endl;
        close(server_fd);
        return -1;
    }

    std::cout << "Server is listening on port 8080..." << std::endl;
    struct sockaddr_in client_addr;
    //创建线程池
    ThreadsPool tp(5);
    //保持循环
    while(1){
        //创建客户套接字
        socklen_t addr_len = sizeof(client_addr);
        //进行信息的绑定， 该方法是一个阻塞方法， 等待一个传入请求
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd < 0) {
            std::cerr << "Error accepting connection." << std::endl;
            close(server_fd);
            return -1;
        }
        //传入进行操作, 这个参数又叫做文件描述符，文件描述符对应的是一个打开的资源
        //底层实现中，每个套接字映射一个文件描述符号， 另外套接字和监听套接字是不一样的
        //也就是说需要对每个线程分配一个套接字进行服务

        // 获取客户端 IP 地址
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, sizeof(ip_str));

        std::cout << "Connected to client with IP: " << ip_str << std::endl;

        //传递套接字进行服务
        tp.enqueue(operate_in_client_server, &client_fd,std::string(ip_str));
        //operate_in_client_server(&client_fd);//单一线程的处理方式

    }

    //关闭套接字

    
}


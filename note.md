首先关于linux下报错的问题可以放在

    打开命令面板（按 Ctrl + Shift + P）。
    输入并选择 "C/C++: Edit Configurations (UI)"，或者直接打开 .vscode/c_cpp_properties.json 文件。
    在 includePath 中添加 test.h 的路径。

进行处理

## 关于cmake怎么使用

我比较习惯这样干， 将文件分成src， build， include， 然后再新建一个CMakeLists.txt

然后在txt文件里面这样子写

    # 指出最低的cmake版本， 不过这个倒是无所谓
    cmake_minimum_required(VERSION 3.10)

    # 设置项目名称和版本
    project(MyProject)

    # 指定 C++ 标准
    # set其实就是设置变量
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    # 添加包含目录, 将从这个目录中寻找头文件
    include_directories(include)

    # 添加源文件， 并且用SOURCES来为其命名， 这样是为了给一组源文件配置对应的内容
    set(SOURCES
        src/main.cpp
        src/test.cpp
    )

    # 根据指定的源文件， 创建名为MyExecutable
    add_executable(MyExecutable  ${SOURCES})

随后，进入build文件， 进行cmake ..指令， 这样是为了在上一级目录中进行cmake操作， 然后进行make编译即可

CMakeLists.txt又被叫做构建规则

后面需要第三方库的导入和管理的时候， 可以使用一个叫vcpkg的工具， 类似pip


## 需要完成的部分

stdio头文件是负责管理输入和输出的函数库

套接字：socket

常见操作如下：

    创建套接字：使用 socket() 函数创建一个套接字。
    int sockfd = socket(AF_INET, SOCK_STREAM, 0); // 创建一个 TCP 套接字

    绑定套接字：将套接字与特定的地址和端口绑定，使用 bind() 函数。
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));

    监听连接（服务器端）：使用 listen() 函数监听客户端的连接请求。

    listen(sockfd, SOMAXCONN); // 监听连接

    接受连接（服务器端）：使用 accept() 函数接受客户端的连接。
    int client_sock = accept(sockfd, NULL, NULL); // 接受连接

    发送和接收数据：使用 send() 和 recv() 函数在套接字上发送和接收数据。
    send(client_sock, message, sizeof(message), 0); // 发送数据
    recv(client_sock, buffer, sizeof(buffer), 0);    // 接收数据

    关闭套接字：使用 close() 函数关闭套接字，释放资源。
    close(sockfd);

实现网络初始化的函数： 参数为端口
参数使用引用， 防止传入为0的时候， 就自动分配一个可用的端口  （tinyhttpd？）
端口为0-65532 所以应该使用无符号的short类型
返回一个套接字

创建套接字， 设置套接字属性， 绑定套接字和网络地址
网络套接字和文件套接字， 分别对应网络通信和本机不同程序的通信
数据流和数据报两种格式的数据
套接字创造失败应该打印相关信息

打印错误原因不能直接使用cout， perror？ 然后结束程序

端口可复用性？（端口复印）

每次获取完套接字以后的属性都要加上判断是否获取正确了

还需要实现的一点是动态端口， 也就是如果当前申请的端口不可用， 就自动获取一个新的本机可用的端口

TCP协议上规定， 每次完成一次活动， 都要关闭tcp链接， 有点奇怪

## 创建多个线程

通过套接字来接待访问， 使用阻塞apiaccpet+循环等待用户的访问

新的套接字只对新的用户进行访问， 原本的套接子可以算作总套接字， 新的套接字可以叫做客户端套接字

但是这样只能呢个进行一次， 要建立多个线程， 一个客户套接字对应一个线程

新的线程里面执行逻辑

## 线程内部执行数据

处理方法里面的参数是套接字

一个请求网页的过程分为两步， 第一步是请求文本， 第二步是根据文本内的资源请求css。js， 图片等等资源

客户端发来的东西是报文， 报文需要一定的格式才能继续被使用， 所以要手动进行解析（真就是解析一行一行的东西）

解析的时候要注意是否同意对应的方法

完成对应的工作以后链接需要关闭

## 资源的管理

实现资源的上传和下载

实现网页请求图片资源的效果

（可选）网页播放视频

## 最后

一些对于多线程的操作

拓展， 压力测试和高并发(epoll)

# 多线程编程的问题

## 进程和线程， 以及线程库thread的操作

进程就是正在运行的程序，进程包含线程

thread std::thread(方法， 数据)

主线程结束的时候，子线程就会结束， 所以要使用thread.join(), 这是一个阻塞方法， 会检测子线程是否return， 暂时阻塞主程序
//另外还有一个办法是分离线程detach
第二个参数是线程函数的参数

## 互斥量

多个线程访问同一个变量，并且对其进行写操作， 那么就要加上一个变量

std::mutex mtx;

使用互斥锁， unlock和lock两个api进行调用， 对数据的写操作加锁

死锁问题： 双方都占有对方的资源， 而且双方都需要对方的资源才能继续下去


## 两个互斥类 lock_guard and unique_lock

两个互斥量的封装类， 实现自动的加锁和解锁， 防止因为竞争互斥量而导致死锁

离开作用区域自动解锁

```
#include <thread>
#include <mutex>

std::mutex mtx;

void print_numbers(int id) {
    lock_guard<std::mutex> lock(mtx); // 进入临界区
    for (int i = 0; i < 5; ++i) {
        std::cout << "Thread " << id << ": " << i << std::endl;
    }
} // 离开作用域自动解锁, 所以一定要加上一个大括号实现局部作用与

int main() {
    std::thread t1(print_numbers, 1);
    std::thread t2(print_numbers, 2);
    
    t1.join();
    t2.join();
    
    return 0;
}
```

## 生产者消费者模型以及条件变量

生产者负责往任务队列中增加任务
消费者负责取出任务， 消费者下面有多个线程用来完成任务。
任务队列为空的时候， 消费者进行等待

condition_variable;

等待条件: 线程在某个条件不满足时，调用条件变量的 wait() 方法进入等待状态。
通知其他线程: 当条件发生变化时，调用 notify_one() 或 notify_all() 来唤醒等待的线程。

wait的第一个参数是一个unique_lock<std::mutex>， 代表在调用的时候持有的锁。
第二个参数是条件， 如果调用的时候是true， 线程就会继续进行。

如果阻塞住了，就需要等待这个条件变量等待notifyone或者all


## 使用线程池

线程池需要维护很多线程， 避免了线程的频繁创建和销毁

生产者加入任务
线程数组从任务队列中获取任务进行执行

## 原子操作

就是原子量，互斥量


# 多路复用

多线程需要设计到上下文的切换，所以要使用单线程的处理方式

每一个网络链接都是一个文件描述符
轮询文件操作符集合，查看是否有数据，这是最简单的办法

## select函数
创建文件描述符号并且保存在数组中
select函数是一个阻塞方法，会根据文件描述符的内容保存一个bitmap，将文件描述符的位置放置为1,其他位置为0
比如 1 2 5 7 9 这五个文件描述符号，那么维护的bitmap就是
01100010101 000000...
select中有五个参数，其一个是最大检索范围，一般是最大的文件描述符+1,二三四都是文件描述符数组，分为读入描写什么的，第五个是最大延长时间
select是一个阻塞方法，当监听到某个文件描述符有动作的时候就会继续进行代码
下面就可以遍历文件描述符，从而进行对应的操作（为什么不直接监听对应的文件操作符号，因为可能一次有多个文件操作符发生变化）
另外注意，select的判断是内核态完成的工作

代码可以理解为这样

```
//内核里面差不多就是发生这样的事情，不过注意不是立即发生阻塞取消，而是遍历完一遍以后发生的取消阻塞
while(1){
    for(auto& fd : fds){
        if(fd发生数据变化)
            //不再阻塞
    }
}
```
但是缺点就是bitmap每次都要设0,并且有上限
除此意外，用户态和内核态切换仍然需要开销

```
#include <iostream>
#include <sys/select.h>
#include <unistd.h>
#include <string.h>

int main() {
    int fd1 = 0; // 假设这是一个有效的文件描述符
    fd_set readfds; //文件描述符集合，这是一个bitmap

    // 初始化文件描述符集合
    FD_ZERO(&readfds);
    // 将文件描述符和文件描述符集合进行绑定
    FD_SET(fd1, &readfds);

    // 调用 select
    int activity = select(fd1 + 1, &readfds, nullptr, nullptr, nullptr);

    //检查文件描述符在文件描述符集合中的位置是否被设置为1了，如果是，则进行后续的操作，从文件描述符中进行读取
    if (FD_ISSET(fd1, &readfds)) {
        std::cout << "Data is available to read." << std::endl;
    }
    

    return 0;
}
```


## poll的东西
poll仍然是阻塞函数，要使用一个名为pollfd的数据结构，其内部包含三个属性
```
struct pollfd{
    int fd;         //文件操作符
    string event;   //event数值为固定字段，用来表示进行何种动作
    srring revent;  //初始的时候为0,如果被修改了，则设置为一个数字
}
```
但是设置位置是revent，第一个参数为pollfd数组（这是一个专门的内置对象）
但是设置位置以后要将revent恢复为0
然后进行读取，本质核心和select差不多

```
#include <iostream>
#include <poll.h>
#include <unistd.h>

int main() {
    //建立一个pollfd数组，pollfd结构体内含三个属性，fd，event，revent
    struct pollfd fds[1];

    //设置文件描述符以及相关的动作
    fds[0].fd = 0; // 假设这是一个有效的文件描述符（如标准输入）
    fds[0].events = POLLIN; // 监视可读事件
    int timeout = 5000; // 5秒超时

    //poll同样是阻塞方法，原理和select是一样的
    int activity = poll(fds, 1, timeout);

    //判断一下是否发生了相关的事件，需要检查一下revent
    if (fds[0].revents & POLLIN) {
        std::cout << "Data is available to read." << std::endl;
    }


    return 0;
}
```

## 关于epoll
生成epfd参数，创建一个白板
使用epoll_ctl绑定epfd，这个时候epfd内部保存了fd以及fd对应的事件
epoll_wait函数
有数据事件的fd会被放到前排
并且会有一个返回值，表示有多少个fd接收到了事件

水平触发和垂直触发

```
#include <iostream>
#include <sys/epoll.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    //创建白板
    int epoll_fd = epoll_create1(0);

    struct epoll_event ev;
    ev.events = EPOLLIN; //设置要做的事情
    ev.data.fd = 0; // 假设这是一个有效的文件描述符（如标准输入）

    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, 0, &ev) ; 

    //设置epoll_event数组（实际上其实应该是很多，但我只弄了一个,并且epoll_event对象塞进去;
    struct epoll_event events[1];
    events[0]=std::move(ev);
    int timeout = 5000; // 5秒超时

    //阻塞方法，等待，并且会将数组进行重新排列，有数据的fd的会放到数组前面
    //返回值为发生事件文件修饰符的数目
    int activity = epoll_wait(epoll_fd, events, 1, timeout);

    //进行数据的读取，实际上这里应该遍历一下数组，从0到activity
    if (events[0].data.fd == 0) {
        std::cout << "Data is available to read." << std::endl;
    }

    //关闭白板
    close(epoll_fd);
    return 0;
}
```
关于事件怎么发生， 一共有两种触发方式，水平触发和边缘触发
水平触发指的是，当文件描述符符合对应状态的时候，就算做是发生事件
边缘触发指的是，只有文件描述符号发生变化的时候才会触发

# 智能指针

在超出作用域，发生异常的时候，会自动回收资源，而普通的指针需要delete释放空间，并且遇到一场也会导致空间无法回收
并且具有所有权这一概念

智能指针一共有三种类型

unique_ptr
所有权只能属于一个指针，需要move移动
独一无二的指针，可以使用move关键字进行移动
```
std::unique_ptr<int> ptr1(new int(5));
std::unique_ptr<int> ptr2 = std::move(ptr1); // 转移所有权
```

shared_ptr
所有权可能属于多个指针
多个共享同一对象，使用引用计数器来计算
多个所有者可以共享一个指针
```
std::shared_ptr<int> ptr1(new int(5));
std::shared_ptr<int> ptr2 = ptr1; // 共享所有权
```

weak_ptr
不占所有权和引用数目
配合shared_ptr使用，但是不计数，被用在非拥有性的对象上面
```
std::shared_ptr<int> ptr1(new int(5));
std::weak_ptr<int> weakPtr = ptr1; // 不增加引用计数
```

可以使用make_unique函数和make_shared直接创建

auto ptr1 = std::make_unique<int>(5); // std::unique_ptr
auto ptr2 = std::make_shared<int>(10); // std::shared_ptr

可以使用*获取数值

int a = *ptr1;

可以使用get函数获取原本指针

int * point = ptr.get();

# cmake的基本使用

是基于CMakeLists.txt作为一个配置文件

常用字段如下

```
//指定cmake版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(MyProject)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加包含目录, 将从这个目录中寻找头文件，一般是直接使用include
include_directories(include)

# 添加源文件，也就是需要进行编译的文件
set(SOURCES
    src/main.cpp
    src/test.cpp
    src/operatehttp
)

# 根据源文件生成生成可执行文件
add_executable(MyExecutable  ${SOURCES})
```

# 对可执行程序使用gdb

gdb target 就是运行可执行文件

run继续执行
quit退出

b/break +行或者函数，在对应的位置打断点

list查看行数

info b 查看断点数目

n/next 继续往下调试

p 打印程序中变量的内容

step 进入某个具体的函数进行调试

watch variable_name 监视某个变量发生改变

core查看崩溃的原因，使用方式为 gdb target core

# 常用bash命令

## 文件和目录操作：
rm：删除文件或目录（使用 -r 选项递归删除目录）。
cp：复制文件或目录。
mv：移动或重命名文件或目录。

## 查看和编辑文件：
cat：显示文件内容。
less：分页查看文件。
nano 或 vim：文本编辑器，用于编辑文件。

## 权限和所有权管理：
chmod：更改文件或目录的权限。
chown：更改文件或目录的所有者。

## 系统信息：
top：实时显示系统资源使用情况。
df：显示文件系统的磁盘空间使用情况。
free：显示内存使用情况。

## 环境变量和配置：
export：设置或显示环境变量。
echo：在终端中输出文本或变量值。

## 进程管理：
ps：显示当前运行的进程。
kill + 进程号：终止运行的进程。

## 网络操作：
ping：测试与主机的网络连接。
curl：发送 HTTP 请求

## 其它用法

command 1 | command 2 将第一个指令的输入作为第二个指令的输出

ps aux | grep bash 查看所有进程并且提取包含bash的进程 grep是抓取的意思

> 输入到某个文件中

# valgrind

用来检查内存是否泄露的东西


# 序列化可以使用第三方库函数

```

struct Person {
    std::string name;
    int id;
    std::string email;
};

nlohmann::json to_json(const Person& person) {
    return nlohmann::json{{"name", person.name}, {"id", person.id}, {"email", person.email}};
}

Person from_json(const nlohmann::json& j) {
    Person person;
    person.name = j.at("name").get<std::string>();
    person.id = j.at("id").get<int>();
    person.email = j.at("email").get<std::string>();
    return person;
}
```

# 常见的http方法

1. GET
用途：请求访问指定的资源，通常用于获取数据。
返回体：应返回请求的资源，通常是 JSON、XML 或 HTML 格式的数据。
状态码：
200 OK：请求成功，返回数据。
404 Not Found：请求的资源不存在。
2. POST
用途：向服务器提交数据，通常用于创建新资源。
返回体：可以返回新创建资源的详细信息，或者返回操作结果的状态。
状态码：
201 Created：成功创建资源，返回新资源的 URI。
400 Bad Request：请求格式错误。
500 Internal Server Error：服务器处理请求时出错。
3. PUT
用途：更新指定的资源，通常用于替换资源的全部内容。
返回体：可以返回更新后的资源或操作结果的状态。
状态码：
200 OK：成功更新资源，返回更新后的数据。
204 No Content：成功更新资源，但不返回数据。
404 Not Found：要更新的资源不存在。
4. PATCH
用途：部分更新指定的资源，通常用于修改资源的部分内容。
返回体：可以返回更新后的资源或操作结果的状态。
状态码：
200 OK：成功更新资源，返回更新后的数据。
204 No Content：成功更新资源，但不返回数据。
404 Not Found：要更新的资源不存在。
5. DELETE
用途：请求删除指定的资源。
返回体：通常返回操作结果的状态，可能不需要返回数据。
状态码：
204 No Content：成功删除资源，但不返回数据。
404 Not Found：要删除的资源不存在。
6. HEAD
用途：与 GET 方法类似，但只请求响应头，不返回响应体。
返回体：无（只返回响应头）。
状态码：
200 OK：请求成功，返回响应头。
404 Not Found：请求的资源不存在。
7. OPTIONS
用途：请求支持的 HTTP 方法，通常用于 CORS 预检。
返回体：返回支持的 HTTP 方法列表。
状态码：
200 OK：请求成功，返回允许的方法。
204 No Content：成功，但不返回数据。

关于图片，js和css等其他资源的请求，都是根据路径重新发送一次http的get请求
文件下载也是一样的方法

# 新安装库

## 包管理工具vcpkg
使用指令为vcpkg install xxxx

## json库函数
下载了一个hpp文件放进include里面了

# RESTfulAPI

重点就是以资源为导向，通过get post put delete等方法进行操作，并且返回对应的状态码

# 关于日志的等级记录

## 需求分析

大致需要完成这些功能

```
1. 访问日志（Access Logs）
请求时间：请求被接收的时间戳。
客户端 IP 地址：发起请求的客户端 IP 地址。
请求方法：如 GET、POST、PUT、DELETE 等。
请求 URL：请求的资源路径。
HTTP 协议版本：如 HTTP/1.1、HTTP/2 等。
响应状态码：如 200（成功）、404（未找到）、500（服务器错误）等。
响应大小：返回给客户端的字节数。
用户代理：客户端浏览器或应用程序的信息。
来源 URL：如果适用，表示请求来源的 URL。

2. 错误日志（Error Logs）
错误时间：发生错误的时间戳。
错误级别：如警告、错误、致命错误等。
错误消息：详细的错误描述信息。
堆栈跟踪：在发生异常时，记录堆栈信息以帮助调试。
请求信息：如果错误与特定请求相关，记录相关的请求信息（如 URL 和方法）。

3. 性能日志（Performance Logs）
处理时间：每个请求的处理时间，帮助分析性能瓶颈。
数据库查询时间：记录数据库操作所需的时间。
外部 API 调用时间：如果服务器调用了外部服务，记录这些调用的延迟。

4. 安全日志（Security Logs）
登录尝试：成功和失败的登录尝试，包括时间、IP 地址和用户信息。
敏感操作：如用户权限更改、数据删除等操作的记录。
异常活动：检测到的可疑活动或入侵尝试的记录。
```

## 代码实现：

函数1：进行文件写入操作并且对文件进行保护， 参数为文件路径， 字符串（如果文件不存在就创建对应的文件）

函数2：访问日志存储函数， 参数为(请求时间，客户端IP，请求方法，请求url，http协议版本，响应状态)

函数3：错误日志存储函数， 参数为(请求时间，客户端ip，请求方法， 请求url，http协议版本，错误原因)

# 支持https


sudo apt-get install libssl-dev
内涵openssl, 但是仍然没支持上去




## 需要复习的大概就是网络链接和线程池， 以及三种多路复用机制



# 关于文档内容

## 安装SQLiteCpp

在github上下载对应的库， 然后安装到了/usr/local目录下面

### find_package的默认寻找路径为：

这个函数的目的是寻找配置文件​（<PackageName>Config.cmake）或 ​模块文件​（Find<PackageName>.cmake）

寻找环境变量<PackageName>_DIR

一些默认的路径 比如SQLiteCpp这个库，寻找到cmake文件 /usr/local/lib/cmake

比如一些常用的文件， 这样

```
/usr/lib/cmake/
/usr/local/lib/cmake/
/usr/share/cmake/Modules/
/usr/local/share/cmake/Modules/
```

如果想要手动指定
find_package(SQLite3 REQUIRED PATHS /path/to/sqlite3 NO_DEFAULT_PATH)

问题在

CMake Error at CMakeLists.txt:37 (add_executable):
  Target "MyExecutable" links to target "SQLite3::SQLite3" but the target was
  not found.  Perhaps a find_package() call is missing for an IMPORTED
  target, or an ALIAS target is missing?


递归查找文件的方式， 比如查询名称里面携带相关内容的东西
find . -type f -name "*SQLite*"

文件要求需要SQLite3Config.cmake 但是可能linux上apt下载的发行版不包括这个cmake文件



下载 SQLite3 源码：
wget https://www.sqlite.org/2023/sqlite-autoconf-3410200.tar.gz
tar -xzf sqlite-autoconf-3410200.tar.gz
cd sqlite-autoconf-3410200
编译并安装：
./configure --prefix=/usr/local
make
sudo make install
检查 CMake 配置文件：
find /usr/local -name "*SQLite3*.cmake"

总之要重新下载一下就行了
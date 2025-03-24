#include <string>
#include <iostream>
#include <map>
#include <istream>
#include <sstream>
#include <vector>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h> // 用于 close 函数
#include <json.hpp> //用于引入json库函数，这个头文件放在include下面了
#include <jwt-cpp/jwt.h>
#include <chrono>
#include "operatelog.h"

using json = nlohmann::json; //引入json的命名空间

//检查文件是否存在
bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();  // 如果文件存在，返回 true
}

//根据文件路径获取文件内容并且将其转化为二进制字符串
std::string getFileContent(const std::string& path) {
    //访问页面要加上相对路径的./
    std::ifstream file("."+path);
    if (!file) {
        return ""; // 文件未找到
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


//检查token是否过期的方法

bool is_token_expired(const std::string& token) {
    try {
        // 解码 JWT
        auto decoded = jwt::decode(token);

        // 获取过期时间 (exp) 声明
        auto exp_claim = decoded.get_payload_claim("exp");

        // 获取当前时间
        auto now = std::chrono::system_clock::now().time_since_epoch().count();

        // 使用 as_int() 获取过期时间戳
        auto exp_time = exp_claim.as_integer(); //草，原来不是as——int

        // 检查是否过期
        return exp_time <= now;
    } catch (const std::exception& e) {
        std::cerr << "Error decoding token: " << e.what() << std::endl;
        return true; // 如果解码出错，视为过期
    }
}


std::map<std::string,std::string> operate_http_request(std::string req){
    //处理请求：
    std::map<std::string,std::string> map_of_request;
    map_of_request.emplace("test","value");
    std::istringstream stream(req);//先将其转化为流
    std::string buff;//创建缓冲区
    //首先先获取协议，域名和请求
    std::string requestLine;
    std::getline(stream, requestLine); // 解析请求行
    // 解析请求行
    std::string method, uri, version;
    std::istringstream requestLineStream(requestLine);
    requestLineStream >> method >> uri >> version;
    //然后将方法协议和uri分别插入map中
    map_of_request.emplace("method",method);
    map_of_request.emplace("url",uri);
    map_of_request.emplace("version",version);
    while(std::getline(stream,buff)&&buff!="\r"){//读取每一行并且保证不是body前面的一行
        int seperate=buff.find(':');
        map_of_request.emplace(buff.substr(0,seperate),buff.substr(seperate+1));
    }
    std::getline(stream,buff);
    map_of_request["body"]=buff;
    return map_of_request;
};

//根据请求路径组织http回复体
std::string response_get(std::map<std::string,std::string>& req,std::string ip){
    //get方法的目标是传递一些资源，请求体中一般都是一些资源，比如二进制的图片，文件，或者html字段
    //当第一次请求的时候，最好还要带上一些东西比如token，放在Authorization字段中
    //这里引入的是token库函数

    std::string secret = "your_secret_key";
    std::string token = jwt::create()
        .set_issuer("example.com") //指定发送者
        .set_subject("user123")    //制定主题
        .set_expires_at(std::chrono::system_clock::now() + std::chrono::hours(1)) //指定过期时间
        .sign(jwt::algorithm::hs256{secret});
    
    std::cout << "Generated Token: " << token << std::endl;

    // 验证 token
    try {
        auto decoded = jwt::decode(token);
        std::cout << "Token is valid!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Token validation error: " << e.what() << std::endl;
    }


    std::string content = getFileContent(req["url"]);
    std::string response;
    if (content.empty()) {
        // 404 响应
        response = "HTTP/1.1 404 Not Found\r\n";
        response += "Content-Type: text/html\r\n";
        response += "Content-Length: 49\r\n"; // 49 是 "404 Not Found" 的长度
        response += "Authorization "+ token +"\r\n";
        response += "\r\n";
        response += "<html><body><h1>404 Not Found</h1></body></html>";
        access_log(ip,req["method"],req["url"],"404");
    } else {
        // 200 响应
        response = "HTTP/1.1 200 OK\r\n";
        response += "Content-Type: text/html\r\n";
        response += "Content-Length: " + std::to_string(content.size()) + "\r\n";
        response += "\r\n";
        response += content; // 响应体为文件内容
        access_log(ip,req["method"],req["url"],"200");
    }
    return response;

}

std::string response_post(std::map<std::string,std::string>& req,std::string ip){
    std::string response;
    //获取具体的路径
    //根据请求内容body来进行post操作
    //然后返回
    // 如果资源不存在 就返回500
    // 如果json字段有问题，就返回400
    std::string path=req["url"];
    std::string body=req["body"];
    //判断是否是动态字段，如果是动态字段就使用map进行接收，如果不是就使用结构体进行接收

    //使用try——catch抛出异常内容
    try {
        // 尝试解析 JSON
        json j = json::parse(body);

        // 创建一个 std::map 来存储键值对
        std::map<std::string, std::string> myMap;
        // 将 JSON 数据转换为 map，json可以直接遍历
        for (auto& element : j.items()) {
            myMap[element.key()] = element.value(); // 将键和值存入 map
        }
        // 输出 map 内容
        for (const auto& pair : myMap) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
        
        // 处理解析后的 JSON 对象

        //整理response.缺少一个情况判断，如果是500的话就不行
        response = "HTTP/1.1 200 OK\r\n";
        response += "Content-Type: text/plain\r\n";
        response += "Content-Length: 0\r\n";
        response += "\r\n";

        access_log(ip,req["method"],req["url"],"200");



    } catch (json::parse_error& e) {
        // 捕获解析错误
        std::cerr << "解析错误: " << e.what() << std::endl;
        std::cerr << "输入的内容是: " << body << std::endl;
        response = "HTTP/1.1 400 Incorrect Format\r\n";
        response += "Content-Type: text/html\r\n";
        response += "Content-Length: 24\r\n"; // 49 是 "404 Not Found" 的长度
        response += "\r\n";
        response += "error: incorrect format";
        access_log(ip,req["method"],req["url"],"400");
    }

    

    return response;



}
std::string response_delete(std::map<std::string,std::string>& req,std::string ip){
    std::string response;
    //检查用户权限信息：检查用户中的Authorization字段
    std::string token=(req.find("Authorization")!=req.end())?req["Authorization"]:"";
    //检查token字段是否过期//这里需要新增一个方法===========================================================(token方法暂时没有经过验证)
    if(is_token_expired(token)){
        //如果token已经过期
        response = "HTTP/1.1 400 Expired\r\n";
        response += "Content-Type: text/plain\r\n";
        response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
        response += "\r\n";
        response += "error: request expired";
        access_log(ip,req["method"],req["url"],"400");
    }else{
        //如果token没有过期
        std::string path = req["path"]; // 替换为你的目录路径
        std::string filename = "test.txt";        // 替换为你要检查的文件名

        if (fileExists(path+filename)) {
            //删除对应文件
            response = "HTTP/1.1 200 OK\r\n";
            response += "Content-Type: text/plain\r\n";
            response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
            response += "\r\n";
            response += "success";
            access_log(ip,req["method"],req["url"],"200");
        } else {
            std::cout << "File does not exist: " << filename << " in " << path << std::endl;
            //删除对应文件
            response = "HTTP/1.1 404 Not Found\r\n";
            response += "Content-Type: text/plain\r\n";
            response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
            response += "\r\n";
            response += "Not found";
            access_log(ip,req["method"],req["url"],"404");
        }

    }
    //进行删除资源查看

    return response;
}
std::string response_put(std::map<std::string,std::string>& req,std::string ip){
    std::string response;
    //put更新资源
    //检查用户权限信息：检查用户中的Authorization字段
    std::string token=(req.find("Authorization")!=req.end())?req["Authorization"]:"";
    //检查token字段是否过期//这里需要新增一个方法
    if(token=="expired"){
        //如果token已经过期
        response = "HTTP/1.1 400 Expired\r\n";
        response += "Content-Type: text/plain\r\n";
        response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
        response += "\r\n";
        response += "error: request expired";
        access_log(ip,req["method"],req["url"],"400");
    }else{
        //如果token没有过期
        std::string path = req["path"]; // 替换为相关目录
        std::string filename = "test.txt";        // 替换为你要检查的文件名

        if (!fileExists(path+filename)) {
            //新建对应文件

            //构建返回体
            response = "HTTP/1.1 200 OK\r\n";
            response += "Content-Type: text/plain\r\n";
            response += "Content-Length: 8\r\n"; // 49 是 "404 Not Found" 的长度
            response += "\r\n";
            access_log(ip,req["method"],req["url"],"200");
        } else {
            //对应文件已经存在
            response = "HTTP/1.1 201 existed\r\n";
            response += "Content-Type: text/plain\r\n";
            response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
            response += "\r\n";
            access_log(ip,req["method"],req["url"],"201");
        }

    }

    return response;
}
std::string response_head(std::map<std::string,std::string>& req,std::string ip){
    std::string response;

    //检查相关资源
    std::string path = req["path"]; // 替换为相关目录
    std::string filename = "test.txt";        // 替换为你要检查的文件名

    if (fileExists(path+filename)) {
        //检查文件最短更新时间
        //构建返回体
        response = "HTTP/1.1 200 OK\r\n";
        response += "Content-Type: text/plain\r\n";
        response += "Content-Length: 8\r\n"; // 49 是 "404 Not Found" 的长度
        response += "Last-Modified: Wed, 21 Oct 2020 07:28:00 GMT\r\n";
        response += "\r\n";
        access_log(ip,req["method"],req["url"],"200");
    } else {
        //对应文件不存在
        std::cout << "File does not exist: " << filename << " in " << path << std::endl;
        response = "HTTP/1.1 404 Not Found\r\n";
        response += "Content-Type: text/plain\r\n";
        response += "Content-Length: 23\r\n"; // 49 是 "404 Not Found" 的长度
        response += "\r\n";
        response += "Not found";
        access_log(ip,req["method"],req["url"],"404");
    }

    
    return response;
}
std::string response_patch(std::map<std::string,std::string>& req,std::string ip){
    std::string response;
    //对资源进行部分更新的方法
    return response;
}
std::string response_options(std::map<std::string,std::string>& req,std::string ip){
    std::string response;
    //返回所支持的所有方法
    response = "HTTP/1.1 200 OK\r\n";
    response += "Allow: GET, POST, OPTIONS\r\n";
    response += "Content-Length: 0\r\n"; // 49 是 "404 Not Found" 的长度
    response += "\r\n";
    access_log(ip,req["method"],req["url"],"200");
    return response;
}

void operate_http_response(std::map<std::string,std::string>& req,int* client_fd,std::string ip){
    //不对不对首先先获取请求方法
    std::string method=req["method"];
    std::cout<<method<<std::endl;
    std::string response;
    //请求体,不过目前只能处理get请求
    if(method=="GET"){
        //构建完整的response
        response=response_get(req,ip);
        //发送response
        send(*client_fd, response.c_str(), response.size(), 0);

    }else if(method=="POST"){
        //post
        response=response_post(req,ip);
    }else if(method=="DELETE"){
        //delete
        response=response_delete(req,ip);
    }else if(method=="OPTIONS"){
        //options
        response=response_options(req,ip);
    }else if(method=="PUT"){
        //put
        response=response_put(req,ip);
    }else if(method=="PATCH"){
        //patch
        response=response_patch(req,ip);
    }else if(method=="HEAD"){
        //head
        response=response_head(req,ip);
    }else{
        std::cout<<"incorrect method"<<std::endl;
    };
    //首先先判断请求路径
    
}
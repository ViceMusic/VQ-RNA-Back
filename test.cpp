#include <string>
#include <iostream>
#include <map>
#include <istream>
#include <sstream>
#include <vector>

#include <ctime>
#include <iomanip>
#include <fstream>
#include <thread>
#include <mutex>

std::mutex mtx;

//获取当前时间并且转化为年月日小时分钟秒格式
std::string get_time(){
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm *localTime = std::localtime(&now);

    // 使用字符串流格式化时间
    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");

    return oss.str();
}


//在文件中写入数据的方法
void write_in_file(std::string path,std::string row){

    //加上互斥类， 避免互斥锁的竞争
    std::unique_lock<std::mutex> lock(mtx);

    // 创建或打开文件以进行写入（以追加模式打开）
    std::ofstream outFile(path, std::ios::app);

    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }

    // 要写入的字符串
    std::string lineToWrite = "This is a new line.";

    // 写入字符串到文件
    outFile << row << std::endl;

    // 关闭文件
    outFile.close();

    lock.unlock();

}

//写入access
void access_log(std::string ip, std::string method, std::string url, std::string state){
    //好像直接进行拼接就可以吧（）
    std::string request_time=get_time();
    std::string message=request_time+" "+ip+" "+method+" "+url+" "+state;
    write_in_file(message,"./access.log");

}
//写入error
void error_log(std::string ip, std::string method, std::string url, std::string error_reason){
    //
    std::string request_time=get_time();
    std::string request_time=get_time();
    std::string message=request_time+" "+ip+" "+method+" "+url+" "+error_reason;
    write_in_file(message,"./error.log");
}
//写入
void reputy_log(){
    //用作暂时写入，方法暂时不用作处理内容
}

int main(){

    std::cout<<get_time();
    write_in_file("./test.log","测试");
}
#ifndef OPERATEHTTP
#define OPERATEHTTP

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

//获取当前时间并且转化为年月日小时分钟秒格式
std::string get_time();

//在文件中写入数据的方法
void write_in_file(std::string path,std::string row);

//写入access
void access_log(std::string ip, std::string method, std::string url, std::string state);
//写入error
void error_log(std::string ip, std::string method, std::string url, std::string error_reason);
//写入
void reputy_log();
#endif
#ifndef OPERATEHTTP
#define OPERATEHTTP
#include <map>
#include <string>
#include <iostream>

std::string getFileContent(const std::string& path); 
std::map<std::string,std::string> operate_http_request(std::string req);
std::string response_get(std::map<std::string,std::string>& req,std::string ip);
std::string response_post(std::map<std::string,std::string>& req,std::string ip);
std::string response_delete(std::map<std::string,std::string>& req,std::string ip);
std::string response_put(std::map<std::string,std::string>& req,std::string ip);
std::string response_head(std::map<std::string,std::string>& req,std::string ip);
std::string response_patch(std::map<std::string,std::string>& req,std::string ip);
std::string response_options(std::map<std::string,std::string>& req,std::string ip);
void operate_http_response(std::map<std::string,std::string>& req,int* client_fd,std::string ip);

#endif
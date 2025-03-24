cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(MyProject)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置路径
set(CMAKE_PREFIX_PATH "/usr/local/cmake")


# 添加包含目录, 将从这个目录中寻找头文件
include_directories(include)

# 查找 OpenSSL 包
find_package(OpenSSL REQUIRED)

# 找到 cpp-jwt 库
# find_package(cpp-jwt REQUIRED)

# 添加源文件
set(SOURCES
    src/main.cpp
    src/test.cpp
    src/operatehttp.cpp
    src/operatelog.cpp
)

# 生成可执行文件
add_executable(MyExecutable  ${SOURCES})

# 链接 OpenSSL 库
target_link_libraries(MyExecutable OpenSSL::SSL OpenSSL::Crypto)
# 链接 cpp-jwt 库
#target_link_libraries(MyExecutable PRIVATE jwt)


const http = require('http');

const data = {
    key1: "value1",
    key2: "value2",
    key3: "value3"
};

// 配置请求选项
const options = {
    hostname: 'localhost',  // 服务器地址
    port: 8080,             // 服务器端口
    path: '/index.html', // 请求路径
    method: 'GET',         // 请求方法
    headers: {
        'Content-Type': 'text/plain', // 请求头
        'Content-Length': Buffer.byteLength('hello') // 内容长度
    },
    body: JSON.stringify(data) // 将对象转换为 JSON 字符串
};

// 创建请求
const req = http.request(options, (res) => {
    console.log(`状态码: ${res.statusCode}`);

    // 处理响应数据
    res.on('data', (chunk) => {
        console.log(`响应体: ${chunk}`);
    });

    // 响应结束
    res.on('end', () => {
        console.log('响应结束');
    });
});

// 处理请求错误
req.on('error', (error) => {
    console.error(`请求错误: ${error.message}`);
});

// 发送数据
req.write(JSON.stringify(data)); // 发送 "hello"
req.end(); // 结束请求



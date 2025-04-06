const http = require('http');
const fs = require('fs');
const path = require('path');

// server只负责请求相关内容, 发送网页, 不承担其他的功能嗷

const port = 8000;

// 创建 HTTP 服务器
const server = http.createServer((req, res) => {
    // 处理根 URL 请求
    if (req.url === '/') {
        const filePath = path.join(__dirname, 'index.html');

        // 读取 usage.html 文件
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('500 Internal Server Error');
                return;
            }

            // 设置响应头并返回文件内容
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    }else if (req.url === '/usage') {
        const filePath = path.join(__dirname, 'usage.html');

        // 读取 usage.html 文件
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('500 Internal Server Error');
                return;
            }

            // 设置响应头并返回文件内容
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    }else if (req.url === '/zhuye.svg') {
        const filePath = path.join(__dirname, '主页.png');

        // 读取 usage.html 文件
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('500 Internal Server Error');
                return;
            }

            // 设置响应头并返回文件内容
            res.writeHead(200, { 'Content-Type': 'image/png' });
            res.end(data);
        });
    } else {
        // 处理未定义的路由
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
    }
});

// 启动服务器
server.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
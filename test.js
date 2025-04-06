const WebSocket = require('ws');

/**
 * 发送 WebSocket 请求并等待响应
 * @param {string} wsUrl - WebSocket 服务器地址 (e.g., 'ws://localhost:8080')
 * @param {Object} requestData - 要发送的请求对象
 * @param {number} [timeout=5000] - 超时时间（毫秒）
 * @returns {Promise<Object>} - 返回服务器响应的 JSON 对象
 */
/*
async function sendWebSocketRequest(wsUrl, requestData, timeout = 500000) {
  return new Promise((resolve, reject) => {
    // 1. 创建 WebSocket 连接
    const socket = new WebSocket(wsUrl);

    // 2. 设置超时定时器
    const timeoutId = setTimeout(() => {
      socket.close();
      reject(new Error(`WebSocket 请求超时 (${timeout}ms)`));
    }, timeout);

    // 3. 连接建立后发送数据
    socket.on('open', () => {
      console.log('WebSocket 连接已建立');
      socket.send(JSON.stringify(requestData)); // 发送 JSON 字符串
    });

    // 4. 处理服务器响应
    socket.on('message', (data) => {
      clearTimeout(timeoutId);
      try {
        const response = JSON.parse(data.toString());
        socket.close();
        resolve(response);
      } catch (e) {
        socket.close();
        reject(new Error('响应解析失败: ' + e.message));
      }
    });

    // 5. 错误处理
    socket.on('error', (err) => {
      clearTimeout(timeoutId);
      reject(new Error('WebSocket 错误: ' + err.message));
    });

    socket.on('close', () => {
      clearTimeout(timeoutId);
    });
  });
}

// 使用示例
(async () => {
  try {
    const wsUrl = 'ws://8.130.10.95:8080';
    const request = {
        type: 'seq',
        user: "hudhd@email.com",
        body: "ATATATATATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTT"
    };

    console.log('正在发送请求:', request);
    const response = await sendWebSocketRequest(wsUrl, request);
    console.log('收到响应:', response);

  } catch (err) {
    console.error('请求失败:', err.message);
  }
})();*/


const fs = require('fs');

// 配置参数
const serverUrl = 'ws://8.130.10.95:8080'; // 替换为你的WS服务器地址
const userEmail = 'user1@example.com'; // 替换为你的用户邮箱
const fastaFilePath = './1-0.fasta'; // 替换为你的FASTA文件路径

// 创建WebSocket连接
const ws = new WebSocket(serverUrl);

// 读取并编码FASTA文件
function encodeFastaFile() {
  try {
    const fileBuffer = fs.readFileSync(fastaFilePath);
    return fileBuffer.toString('base64');
  } catch (error) {
    console.error('文件读取失败:', error);
    process.exit(1);
  }
}

// 构造消息对象
const requestPayload = {
  type: "user_tasks",  // 使用文件模式
  email: 'user1@example.com',
};

// 连接事件处理
ws.on('open', function open() {
  console.log('已连接到服务器');

  // 发送请求数据
  ws.send(JSON.stringify(requestPayload), (err) => {
    if (err) console.error('发送失败:', err);
  });
});

// 接收响应处理
ws.on('message', function incoming(data) {
  try {
    const response = JSON.parse(data);

    // 处理不同类型的响应
    if (response.status === 'success') {
      console.log('✅ 请求成功:', response.message);
      if (response.data) {
        console.log('响应数据:', response.data);
      }
    } else if (response.status === 'error') {
      console.error('❌ 服务器返回错误:', response.error);
    } else {
      console.warn('⚠️ 未知响应格式:', response);
    }

    // 关闭连接
    ws.close();
  } catch (e) {
    console.error('响应解析失败:', e);
  }
});

// 错误处理
ws.on('error', function error(err) {
  console.error('连接错误:', err);
});

ws.on('unexpected-response', (req, res) => {
  console.error('意外响应:', res.statusCode);
});

// 关闭连接处理
ws.on('close', function close() {
  console.log('连接已关闭');
});
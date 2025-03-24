//g++ program.cpp -o program && ./program && rm program

const { exec } = require('child_process');

exec('g++ test.cpp -o program && ./program && rm program', (error, stdout, stderr) => {
    if (error) {
        console.error(`执行错误: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`标准错误: ${stderr}`);
        return;
    }
    console.log(`标准输出: ${stdout}`);
});
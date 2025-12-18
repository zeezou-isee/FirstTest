1. 在Windows安装wsl系统
    0. 打开Windows相关功能：
        0.1. 在控制面板 -> 程序 -> 启用或关闭windows功能
        0.2. 勾选"适用于 Linux 的 Windows 子系统"和"虚拟机平台"
        0.3. 重启windows
    1. 管理运行powershell，输入：
        ```
        wsl --list --online
        ```
    2. 安装指定版本Ubuntu:
        ```
        wsl --install -d <Distribution Name>
        ```

2. 安装adb工具
    2.1. download from link: https://dl.google.com/android/repository/platform-tools-latest-windows.zip
    2.2. unzip this file into disk: e.g. "D:/adb"
    2.3. add the folder(e.g. "D:/adb") into system path.

3. 在wsl系统中使用设备
    3.1. 准备数据线，连接android手机到usb接口
    3.2. use instruction: ```adb devices`` to check your phone connection
    3.3. run ```usbipd list``` to check your phone usb port
    3.4. run ```usbipd bind --busid <port> ``` to share your device
    3.5. run ```usbipd attach --busid <port> --wsl <unbutu version> ``` #coonfirm on your phone "using for file transport" 
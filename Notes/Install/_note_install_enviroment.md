# tensorflow learning note
本教程默认使用者有基本的英语阅读能力 , 所以会有很多官方原版英文文档 ,本人英语六级配合google翻译既可阅读 ,不用担心.
## 1. install

1.  install Anaconda  
    [清华源镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)  
    [换源教程](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)  
    [添加环境变量](EnvironmentPathSet.md)
    [pip使用官方教程](https://pip.pypa.io/en/stable/)  
    [pip换源官方教程](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
    尽量使用 *pip* 语句安装;
2.  install tensorflow & keras  
### main install steps
#### step1 :
在**Anaconda Navigator** 中 ,
点击左侧 **Environment** ,  
在右侧新建一个环境(下方 *Create* 选项) ,命名为 *tfenv* (可以自选,但是需要和下文的activate语句 后字符串相同)  
选择python版本为3.X  
(我安装时最高支持版本为3.6 , 具体看官方声明和实际需求)  
**或者**
在Anaconda Prompt 中 ,
使用命令
```
conda create -n 虚拟环境名称  python=版本 anaconda
```
#### step2 :
在Anaconda Prompt 中 ,  
输入:  
    ```
    activate tfenv
    ```
如果要安装CPU版本(显卡不是NIVDIA或不支持CUDA / CUDNN)
    ```
    conda install tensorflow
    ```
如果安装GPU版本(显卡支持CUDA / CUDNN)
    ```
    conda install tensorflow-gpu
    ```
CUDA 和 CUDNN需要自己下载安装 , 安装教程参见[CUDA&CUDNNInstall](CUDA&CUDNNInstall.md)
### Tips
1.  **如果出现找不到库的情况,使用如下语句**  
    **(注意使用activate xxx 切换到你建立的环境中)**  
    ```
    conda install protobuf==3.6.0
    ```
2.  **Tensorflow 2.0 自带keras包 , 不再需要额外安装**  
3.  **关于GPU版本 , 如果出现 找不到cudart64_100.dll 或者cublas64_100.dll 的情况,到CUDA的安装位置分别查找cudart64 和 cublas64 ,将找到的文件文件名部分的'_'后内容修改为100 , 没有出现这个问题则不需要改**

## 2. use tensorflow on VSCode
我使用的是 *VSCode* IDE , *Spyder* 未使用所以没写 , 不习惯的可以自查.
### 具体安装教程  
[VSCode官方下载](https://code.visualstudio.com/)  
[VSCode官方安装教程](https://code.visualstudio.com/docs/supporting/faq)
1.  安装 *Code Runner* 插件;**代码运行支持**
2.  安装 *Chinese (Simplified) Language Pack for Visual Studio Code* 插件;**中文支持** [ *可选* ]
3.  安装 *python* 插件;**python支持**
4.  安装 *Anaconda Extension Pack* 插件;**Anaconda支持**
5.  安装 *MagicPython* 插件;**语法高亮支持**

### 设置
1.  在 *Code Runner* 插件的设置中将 *"python"...* 行修改为
    
        "python" : "set PYTHONIOENCODING=utf8 && $pythonPath $fullFileName"

2.  在运行python程序之前 , 点击下边栏左下角第一个选项 , 选择合适的运行环境(与前述activate 语句激活的环境相同)
3.  Ctrl + Alt + N 既可运行程序 ;
4.  在系统环境变量中新建一个"PYTHONIOENCODING" 变量 , 值为"utf-8" **不是在path里 , 和它平级**

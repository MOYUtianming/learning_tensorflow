**仅以Win10为例 , 其余系统可以自行搜索**
# General set path
1.  打开windows设置

        Win + X组合 -> N键 

2.  搜索  
    
        "环境变量"

3.  选择
    
        编辑系统环境变量

4.  按 N 键或选择弹窗中右下角的"环境变量"
5.  在弹窗中的"系统变量"分栏中双击 
    
        Path

6.  将需要的环境变量的绝对路径添加入新弹窗中,并点击确认

# Anaconda set path details
一共有三个文件夹需要加入系统环境变量中 :  
(下文中的Anaconda为软件安装目录 ,  *".../"*  被我用于省略前置路径,实际使用时需要补全)  

    ...\Anaconda3  
    ...\Anaconda3/Scripts  
    ...\Anaconda3/Library/bin  

三者分别指向 :
默认的python主程序 ,  
Anaconda安装时附带的脚本(Scripts)  
以及 常用工具(bin)  
# Tensorflow set path details
与上文类似 , 一共有三个路径需要添加到系统环境变量中 :

        ...\Anaconda3\envs\tfenv
        ...\Anaconda3\envs\tfenv\Scripts
        ...\Anaconda3\envs\tfenv\Library\bin

其中**tfenv**应当被置换为你之前建立的环境的名称.
# Tensorflow GPU_x64_win
需要添加的路径为

        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\CUPTI\lib64
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\cudnn\cuda\bin
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\libnvvp

# PyKAN

## 1. 功能概述
PyKAN 是一个基于 Kolmogorov-Arnold 表示定理的新型神经网络库，其核心思想是用可学习的激活函数（如样条函数）替代传统神经网络中的固定激活函数和权重，从而构建更具可解释性和高精度的模型。与标准多层感知机（MLP）不同，KAN 网络在边（而非节点）上学习函数，使其能够更灵活地拟合复杂数据模式，在数学函数逼近、科学计算和符号回归等任务中展现出比传统 MLP 更强的表达能力和解释性，为可解释人工智能提供了一种新思路。

- 参考实现：
    ```
    url=https://github.com/KindXiaoming/pykan
    commit_id=ecde4ec3274d3bef1ad737479cf126aed38ab530
    ```

## 2. 安装PyKAN

### 2.1 基础环境安装
请参考[Teco用户手册的安装准备章节](http://docs.tecorigin.com/release/torch_2.4/v2.2.0/#fc980a30f1125aa88bad4246ff0cedcc)，完成训练前的基础环境检查和安装。

### 2.2 构建docker
#### 2.2.1 执行以下命令，下载Docker镜像至本地（Docker镜像包：pytorch-2.2.0-torch_sdaa2.2.0.tar）
    ```
    wget http://wb.tecorigin.com:8082/repository/teco-docker-tar-repo/release/ubuntu22.04/x86_64/2.2.0/pytorch-2.2.0-torch_sdaa2.2.0.tar
    ```
#### 2.2.2 校验Docker镜像包，执行以下命令，生成MD5码是否与官方MD5码b2a7f60508c0d199a99b8b6b35da3954一致：
    ```
    md5sum pytorch-2.2.0-torch_sdaa2.2.0.tar
    ```
#### 2.2.3 执行以下命令，导入Docker镜像
    ```
    docker load < pytorch-2.2.0-torch_sdaa2.2.0.tar
    ```
#### 2.2.4 执行以下命令，构建名为pykan的Docker容器
    ```
    docker run -itd --name="pykan" --net=host --device=/dev/tcaicard0 --device=/dev/tcaicard1 --device=/dev/tcaicard2 --device=/dev/tcaicard3 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 64g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.2.0-torch_sdaa2.2.0 /bin/bash
    ```
#### 2.2.5 执行以下命令，进入名称为pykan的Docker容器。
    ```
    docker exec -it pykan bash
    ```
#### 2.2.6 执行以下命令安装pykan
    ```
    cd <pykan>
    pip install -e .
    ```
- 安装后可执行以下命令验证安装成功
    ```
    python -c "from kan import KAN; print('pykan imported successfully')"
    ```    

### 2.3 功能验证
- 开启sdaa环境
    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate
    ```
- 运行<pykan>/tutorials/Example/路径下的文件测试功能是否正常（可以将其他jupyter文件改写为python文件）
    ```
    python <pykan>/tutorials/Example/test_unsupervised_learning.py
    ```
- 构建：`docker build -t registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023 .`
- 推送：`docker push registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023`
- 运行：` docker run --gpus 1 --rm -it -v /mnt/f/dw/CBC2023CRC/examples/datas/:/mounted registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023 bash run.sh /mounted/in.lst /mounted/out`
  - 没有GPU的话需要把`--gpus 1`参数去掉
  - 如果使用GPU必须确保显卡驱动、`cuda-toolkit`、`nvidia-container-toolkit`都安装了

## 阿里云 docker 使用

### 1. 登录阿里云Docker Registry

```
$ docker login --username=10619*****@qq.com registry.cn-hangzhou.aliyuncs.com
```

用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。

您可以在访问凭证页面修改凭证密码。

### 2. 从Registry中拉取镜像

```
$ docker pull registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023:[镜像版本号]
```

### 3. 将镜像推送到Registry

```
$ docker login --username=10619*****@qq.com registry.cn-hangzhou.aliyuncs.com$ docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023:[镜像版本号]$ docker push registry.cn-hangzhou.aliyuncs.com/sulab/cbc2023:[镜像版本号]
```

请根据实际镜像信息替换示例中的[ImageId]和[镜像版本号]参数。

### 4. 选择合适的镜像仓库地址

从ECS推送镜像时，可以选择使用镜像仓库内网地址。推送速度将得到提升并且将不会损耗您的公网流量。

如果您使用的机器位于VPC网络，请使用 registry-vpc.cn-hangzhou.aliyuncs.com 作为Registry的域名登录。

### 5. 示例

使用"docker tag"命令重命名镜像，并将它通过专有网络地址推送至Registry。

```
$ docker imagesREPOSITORY                                                         TAG                 IMAGE ID            CREATED             VIRTUAL SIZEregistry.aliyuncs.com/acs/agent                                    0.7-dfb6816         37bb9c63c8b2        7 days ago          37.89 MB$ docker tag 37bb9c63c8b2 registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816
```

使用 "docker push" 命令将该镜像推送至远程。

```
$ docker push registry-vpc.cn-hangzhou.aliyuncs.com/acs/agent:0.7-dfb6816
```
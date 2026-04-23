# 开发指南

## 容器镜像

| 硬件型号 | CPU架构 | 容器镜像 | Dockerfile |
|---------|---------|---------|-----------|
| Atlas 800I/T A2 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20251219 | [openeuler_torch_ascend_arm.Dockerfile](../docker/openeuler_torch_ascend_arm.Dockerfile) |
| Atlas 800I/T A2 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20251219 | [openeuler_torch_ascend_x86.Dockerfile](../docker/openeuler_torch_ascend_x86.Dockerfile) |
| Atlas 800I/T A3 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 | [openeuler_torch_ascend_a3_arm.Dockerfile](../docker/openeuler_torch_ascend_a3_arm.Dockerfile) |
| Atlas 800I/T A3 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20260324 | [openeuler_torch_ascend_a3_x86.Dockerfile](../docker/openeuler_torch_ascend_a3_x86.Dockerfile) |

创建容器：

```bash
# 创建容器
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20251219 /bin/bash
```

## 源码安装

进入容器后可使用源码编译安装：

```bash
git clone https://atomgit.com/openeuler/GVirt.git
cd GVirt/xlite
pip install -r requirements-build.txt  # 安装构建依赖（若跳过，后续安装需移除--no-build-isolation参数）
pip install . --no-build-isolation  # 安装当前目录下的xlite包

# 若开发环境安装，建议使用"-v .[dev]"（py源码修改后可直接在开发环境中生效）
pip install -v -e .[dev] --no-build-isolation
```

## 编译

需提前安装构建依赖：

```bash
cd GVirt/xlite
pip install -r requirements-build.txt
pip install -r requirements-dev.txt  # 进一步安装开发依赖（可选）
```

编译步骤：

```bash
# 准备
rm -rf build && mkdir -p build
# 编译
cmake -B build && cmake --build build -j
# 安装
cmake --install build
# 测试验证：可使用算子测试，也可使用完整模型测试
python tests/kernels/add.py
```

## 功能验证
```bash
bash tests/run_accuracy.sh
```

## 性能测试

vllm_ascend + xlite在线服务的性能测试及性能对比分析，请参考 [e2e_test.md](doc/e2e_test.md)


## 代码提交

代码提交前请本地容器环境内执行格式检查，代码检查及修正方法参考：[静态检查说明](static_checker.md)


## 构建安装包

当前支持rpm和whl，可选择合适的方式构建出包，用于不同场景的二进制发布和安装部署。

### 构建rpm包

```bash
# 切换到xlite目录下，执行以下命令准备rpm构建环境
mkdir -p /root/rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SOPES,SPECS,SRPMS}
# 拷贝源码至/root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz，执行以下命令
VERSION=0.1.0  # 替换为当前版本号
git archive --format=tar.gz --prefix=xlite-${VERSION}/ -o /root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz HEAD
cp xlite.spec /root/rpmbuild/SPECS/
cd /root/rpmbuild/SPECS
rpmbuild -bb xlite.spec --nodebuginfo
```

构建后生成的rpm包在`/root/rpmbuild/RPMS/`目录下。

### 构建whl包

推荐方式（完整遵循`pyproject.toml`中的构建配置）：

```bash
python -m build --wheel --no-isolation  # 构建whl包；如需.tar.gz包可去掉--wheel参数
```

传统方式（可能不会自动安装`[build-system]`依赖）：

```bash
python setup.py bdist_wheel && python setup.py clean  # 构建whl包并清理构建产物
```

构建后生成的whl包在`dist`目录下。

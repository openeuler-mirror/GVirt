# 开发指南

## 容器镜像

| 硬件型号 | CPU架构 | 容器镜像 | Dockerfile |
| --------- | --------- | --------- | ----------- |
| Atlas 800I/T A2 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20251219 | [openeuler_torch_ascend_arm.Dockerfile](../../docker/openeuler_torch_ascend_arm.Dockerfile) |
| Atlas 800I/T A2 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20251219 | [openeuler_torch_ascend_x86.Dockerfile](../../docker/openeuler_torch_ascend_x86.Dockerfile) |
| Atlas 800I/T A3 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 | [openeuler_torch_ascend_a3_arm.Dockerfile](../../docker/openeuler_torch_ascend_a3_arm.Dockerfile) |
| Atlas 800I/T A3 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20260324 | [openeuler_torch_ascend_a3_x86.Dockerfile](../../docker/openeuler_torch_ascend_a3_x86.Dockerfile) |

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
pip install -r requirements-build.txt --extra-index https://download.pytorch.org/whl/cpu/  # 安装构建依赖（若跳过，后续安装需移除--no-build-isolation参数）
pip install . --no-build-isolation --extra-index https://download.pytorch.org/whl/cpu/  # 安装当前目录下的xlite包

# 若开发环境安装，建议使用"-v .[dev]"（py源码修改后可直接在开发环境中生效）
pip install -v -e .[dev] --no-build-isolation --extra-index https://download.pytorch.org/whl/cpu/

# 可选：设置 XLITE_BUILD_TEMP 复用稳定的 cmake 构建树，使重复的编辑安装增量编译（wheel/发布构建不受影响，始终全新构建）
XLITE_BUILD_TEMP=~/.cache/xlite/build pip install -v -e .[dev] --no-build-isolation --extra-index https://download.pytorch.org/whl/cpu/
```

若仅需测试正在开发的功能，可直接从主分支安装：

```bash
pip install git+https://atomgit.com/openeuler/GVirt.git@master#subdirectory=xlite --extra-index https://download.pytorch.org/whl/cpu/
```

## 编译

需提前安装构建依赖：

```bash
cd GVirt/xlite
pip install -r requirements-build.txt --extra-index https://download.pytorch.org/whl/cpu/
pip install -r requirements-dev.txt --extra-index https://download.pytorch.org/whl/cpu/  # 进一步安装开发依赖（可选）
```

编译步骤：

```bash
# 准备（可选）: 清理旧的构建目录
rm -rf build && mkdir -p build
# 编译
cmake -B build && cmake --build build -j "${MAX_JOBS:-128}"
# 安装
cmake --install build
# 测试验证：可使用算子测试（稳定集），也可使用完整模型测试
pytest tests/kernels
```

ccache 为可选的编译加速项：cmake 配置阶段自动探测（`find_program`），未安装则不启用，构建不依赖 ccache。复用的构建目录（如 `build/` 或 `XLITE_BUILD_TEMP` 指定的稳定构建树）若在配置后 ccache 被移除，重新配置时会自动校验缓存路径并重新探测，无需手动清理构建目录。

### 调试构建

xlite 的调试能力集中在 `csrc/debug.{h,cpp}` 模块：提供 rank 感知、配色、单次 flush 的原子打印（避免多 rank /多线程输出交错），以及张量检查算子（`XDEBUG_PRINT*` / `XDEBUG_CHECK_NAN_INF` 等，附带 NaN/Inf/大值检测）。调试代码默认不编译，通过 `XLITE_DEBUG_ON` 环境变量按类别开启——这是一个**编译期**开关，仅在 cmake 配置阶段读取，改值后需重新配置并编译。

```bash
# 仅前向中间张量打印（每层 attn/ffn 前后的张量检查）
XLITE_DEBUG_ON=forward cmake -B build && cmake --build build -j "${MAX_JOBS:-128}"
# 前向 + auto_tuner tile size 调试
XLITE_DEBUG_ON=forward,tuner cmake -B build && cmake --build build -j "${MAX_JOBS:-128}"
# 全部调试类别
XLITE_DEBUG_ON=all cmake -B build && cmake --build build -j "${MAX_JOBS:-128}"
```

支持的类别 token（逗号分隔、不区分大小写）及对应编译宏见 [环境变量说明文档](../user_guide/ENVIRONMENT_VARIABLES.md#xlite_debug_on)。机制上，CMake 将该变量解析为编译宏（`XLITE_DEBUG_ON` 基础宏 + 每个 token 的 `XLITE_DEBUG_ON_<TOK>`），具体哪个宏控制哪段调试代码由 `csrc/debug.h` 决定，因此 `forward` 与 `tuner`/`gettensor` 相互独立、可按需组合。

> **wheel 构建说明**：`setup.py` 在非可编辑（wheel）构建时会从 cmake 子进程环境中剥离 `XLITE_DEBUG_ON`，确保发布包不带调试代码。开发可编辑安装（`pip install -e .`）会保留该变量，故可如上携带。

## 内核单元测试（pytest 稳定集）

`tests/kernels/` 中以 `test_*.py` 命名的文件由 pytest 收集，构成"稳定集"；其余为旧式循环脚本（`python tests/kernels/<op>.py` 直接运行），按文件逐个迁移。进入稳定集的标准：单卡可跑、随机输入有种子（失败可复现）、且在 CI 容器上连续多次运行（建议 5 次浸泡）零失败。分布式（torchrun 多卡）用例与存在误报/已知失败的用例不进稳定集，迁移后以 `multi_npu` / `flaky` / `broken` 标记排除在默认运行之外。

```bash
pytest tests/kernels                 # 运行稳定集（默认排除上述三类标记；卡选择同 run.sh，用 ASCEND_RT_VISIBLE_DEVICES）
pytest tests/kernels -m broken       # 选择性运行被标记的用例（显式 -m 覆盖默认过滤）
for i in $(seq 5); do pytest tests/kernels || break; done   # 迁移入集前的浸泡验证
```

单文件迁移清单（`op.py` -> `test_op.py`）：

1. `git mv tests/kernels/op.py tests/kernels/test_op.py`；
2. 模块级 `Runtime(...)` / `set_device` 改用 `tests/kernels/conftest.py` 的会话级 `runtime` fixture（`allow_internal_format` 与生产 model runner 一致地设好；显存池可用 `XLITE_TEST_RUNTIME_MB` 调整）；随机输入由 `reproducible_rng` fixture 按用例自动播种；`default_device` fixture 已把默认张量设备设为 `npu:0`，工厂调用省略 `device=`（CPU 参考张量需显式 `device="cpu"`）；
3. 循环枚举改为 `@pytest.mark.parametrize`（生成可读 id）；比较失败直接抛出，不再 `try/except` 吞掉；
4. 保留 `if __name__ == "__main__": raise SystemExit(pytest.main([__file__, "-q"]))` 尾部，直接执行（`python tests/kernels/test_op.py`）与 `tests/performance/perf_xlite_ops.py` 的 msprof 包装仍可用；
5. 浸泡验证通过后，更新 docs/注释中的旧路径引用再提交。

## 功能验证

```bash
bash .gitcode/run_accuracy.sh
```

## 张量布局语义

`csrc/base.h` 中 `XTensor` 的 shape/stride/bytes/连续性语义(含 torch 张量包装规则与非连续
张量的限制)见 [xtensor.md](xtensor.md)。

## 性能测试

vllm_ascend + xlite在线服务的性能测试及性能对比分析，请参考 [e2e_test.md](e2e_test.md)

## 代码提交

代码提交前请本地容器环境内执行格式检查，代码检查及修正方法参考：[静态检查说明](static_checker.md)

## 构建安装包

当前支持rpm和whl，可选择合适的方式构建出包，用于不同场景的二进制发布和安装部署。

### 构建rpm包

```bash
# 切换到xlite目录下，执行以下命令准备rpm构建环境
mkdir -p /root/rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SOPES,SPECS,SRPMS}
# 从当前 git tag 重新生成 xlite.spec（Version 与 wheel 的 setuptools_scm 版本同源）
python scripts/make_spec.py
VERSION=$(awk '/^Version:/{print $2}' xlite.spec)
# 拷贝源码至/root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz，执行以下命令
git archive --format=tar.gz --prefix=xlite-${VERSION}/ -o /root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz HEAD
cp xlite.spec /root/rpmbuild/SPECS/
cd /root/rpmbuild/SPECS
rpmbuild -bb xlite.spec --nodebuginfo
```

构建后生成的rpm包在`/root/rpmbuild/RPMS/`目录下。

### 构建whl包

推荐方式（完整遵循`pyproject.toml`中的构建配置）：

```bash
pip install build  # 安装build工具
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ python -m build --wheel --no-isolation  # 构建whl包；如需.tar.gz包可去掉--wheel参数
```

传统方式（可能不会自动安装`[build-system]`依赖）：

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ python setup.py bdist_wheel && python setup.py clean  # 构建whl包并清理构建产物
```

构建后生成的whl包在`dist`目录下。

## 代码检查

代码检查基于 [pre-commit](https://pre-commit.com/)（配置见 `xlite/.pre-commit-config.yaml`，仓库根目录的 `.pre-commit-config.yaml` 为其符号链接），覆盖 Python、C/C++、Markdown 与拼写检查：

```bash
pre-commit install --install-hooks                # 一次性安装并预构建全部 hook 环境，之后每次 commit 自动检查
pre-commit run --all-files                        # 提交态全树快检（无需 CANN 构建树/mypy/clang-tidy；无 node 环境时前置 SKIP=markdownlint）
python .gitcode/run_static_checks.py              # 提交 PR 前运行全部检查（与 CI 一致）
```

详见 [静态检查说明](static_checker.md)。

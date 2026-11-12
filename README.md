# 介绍

DLCompiler是上海人工智能实验室（上海 AI 实验室）DeepLink 团队开源扩展 Triton 的深度学习编译器：

- 跨架构 DSL 扩展：通过扩展 DSL，让 DSA 芯片（昇腾芯片）也能享受 GPU 级的编程体验和性能，成为 “跨架构 AI Kernel DSL” 。
- 智能自动优化：实现智能核间调度，充分释放多核算力；结合创新的访存合并优化，将离散访问自动重组为高速连续访问，大幅提升算子性能与带宽利用率。
![身位图-昇腾](https://github.com/user-attachments/assets/59c195cc-2702-4d5a-8559-3bed1722281e)

## 基于华为昇腾芯片

### 支持芯片

910B2

### pip安装

```bash
# 因包过大，超过pypi限制（我们也在申请更大的容量），暂时支持从github安装
pip install https://github.com/DeepLink-org/DLCompiler/releases/download/v0.0.2/dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
# 也可以先下载，然后再安装
wget https://github.com/DeepLink-org/DLCompiler/releases/download/v0.0.2/dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
pip install dlcompiler-3.4.0-cp310-cp310-linux_aarch64.whl
```

### 源码安装

#### 安装ascend cann

1. 要求CANN 版本 >= 8.3.RC1
2. 社区下载链接：[下载链接](https://www.hiascend.com/developer/download/community/result?module=cann)
3. 社区安装指引链接：[下载链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)

#### 安装依赖

```bash
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1  \
    psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11 nanobind
```

#### 安装torch_npu

1. 要求torch_npu版本 >= 2.6.0

```bash
pip install torch_npu==2.6.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

#### 编译命令

```bash
bash compile_shared.sh apply_patch=true     # 如果不应用patch，可以直接执行 bash compile_shared.sh
```

### 查看编译过程的mlir文件

```bash
export TRITON_DEBUG=1 # 默认在当前目录下
```

### 测试

```bash
python ./test/ascend/passed_tests/test_silu_and_mul.py
```

## 基于寒武纪芯片

### 支持寒武纪芯片

MLU

### 寒武纪上编译

```bash
bash compile_on_mlu.sh
```

### 寒武纪测试

```bash
cd build/triton/tutorials
python 01-vector-add.py
```

## 基于沐曦芯片

### 支持沐曦芯片

C500

### 沐曦上编译

```bash
bash compile_on_maca.sh
```

### 沐曦自定义算子（Custom Op）

DLCompiler 在沐曦后端同样支持 `dl.custom()` 自定义算子。编译流程为：

`dl.custom()` → `triton_gpu.custom`（携带 Layout 与调度属性）→ `tt.extern_elementwise`（调用用户 bitcode 符号）→ 链接用户 bitcode

使用方式（完整示例见 `test/maca/test_custom_op.py`）：

```python
import triton.language.extra.deeplink.cann.extension as dl

@dl.register_custom_op
class add:
    core = dl.CORE.MXU          # MACA 核类型：MXU / SIMT / TENSOR
    pipe = dl.PIPE.MX_COMPUTE   # MACA 流水线：MX_PIPE / MX_COMPUTE / MX_MEMORY
    mode = dl.MODE.MX_SIMD      # MACA 执行模式：MX_SIMD / MX_SIMT / MX_TENSOR

    def __init__(self, a, b, out=None):
        assert out is not None, "dl.custom() requires out= parameter"
        self.symbol = "custom_add_" + str(a.dtype)
        self.bitcode = "maca_add"  # 自动解析为 maca_add.bc

@triton.jit
def custom_add_kernel(output_ptr, a_ptr, b_ptr, L: tl.constexpr):
    idx = tl.arange(0, L)
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    buf = tl.full([L], 0, a.dtype)
    res = dl.custom("add", a, b, out=buf)
    tl.store(output_ptr + idx, res)
```

bitcode 中的函数按"逐元素"ABI 实现（编译器为张量的每个元素生成一次调用，
bitcode 内无需处理线程组织），例如 `custom_add_fp32(float a, float b)`。
示例源码见 `dlcompiler/bitcode/src/maca_add.cc`，编译：

```bash
# 生成 language/deeplink/bitcode/bc/maca_add.bc（依赖 MACA_PATH 或 PATH 中的 mcclang/clang）
bash dlcompiler/bitcode/compile_maca_bc.sh

# 运行测试（需沐曦设备）
python test/maca/test_custom_op.py
```

也可通过 `DLCOMPILER_BITCODE_PATH` 添加 bitcode 搜索路径，或在注册算子时
直接传入 `.bc` 文件的绝对路径。

# TRICO-R-PINO / pure-PINO

外部条件驱动的湖泊温度场直接神经算子研究工程。当前发布基于
`LayerNorm_split_SI_clock` 最新候选，保留原生热力学、冰雪、表面与床热接口。
它是**尚未训练的新候选代码**；示例和软件检查不代表真实湖泊精度验收。

This repository contains an editable Python package for the current direct
lake-field candidate, with explicit SI clock contracts and the original native
physics functions. Temperature is decoded by a neural field; the old 900-second
temperature integrator is not included in the primary forward.

## 在 VS Code 打开

```text
git clone git@github.com:Quan05-TP/TRICO-R-PINO.git
cd TRICO-R-PINO
code TRICO-R-PINO.code-workspace
```

首次打开接受推荐扩展，选择项目的 Python 环境。Python 3.12 是本工程的开发基线，
Python 3.10 及以上可安装。已有科学运行环境请保留，在独立环境中开发。

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pure_pino doctor
python -m pure_pino example
python -m pytest
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pure_pino doctor
python -m pure_pino example
python -m pytest
```

`doctor`检查包、最新时间合同与真实原生源码哈希；`example`使用明确标注的制造外部输入，
展示初始化、24小时块解码、carry、真实物理解码和梯度。它不读取目标温深、不下载数据，
也不启动训练。`example`需从克隆仓库以editable方式安装；独立wheel只提供库与`doctor`。
需要CUDA时，另按所在服务器已验证的PyTorch环境安装；CPU CI不配置CUDA。

## 模型入口

```python
from pure_pino import create_model, native_adapter

model = create_model()  # 明确选择 split_native_support_affine，未训练
adapter = native_adapter()  # 四个原生纯函数，SHA绑定
```

batch、`CoupledExternal`和完整查询示例见
[examples/external_field.py](examples/external_field.py)。输入单位和边界见
[数据接入说明](docs/DATA_INTEGRATION.md)。直接构造`DirectPhaseModelConfig`时，旧配置默认仍用于
legacy兼容；新实验应使用`create_model`或显式声明新时间合同。

```text
气象前史 + 独立几何 / 合格外部观测
                 ↓
时间与气象分离编码 + 注意力 + 几何图 / Fourier核
                 ↓
学习初态 → 直接时间块解码 → 学习状态传递
                 ↓
原生温度 / 焓 / 冰雪 / 床热读出与物理残差
```

## 目录

```text
src/pure_pino/       可编辑的模型、时钟、相态与物理接口
  _features/        原绑定版本的冻结特征依赖
  _native/          四个原生纯函数，保留原字节和哈希
examples/           仅外部输入的接口示例
configs/            示例配置
tests/              软件、信息边界、梯度和原生函数回归
docs/               数据接入、科学状态、来源与协作说明
.vscode/            调试、测试、格式检查和推荐扩展
.devcontainer/      可选统一CPU开发容器
.github/            CI、PR与Issue模板
```

格式与检查：

```text
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

`_features`和`_native`保留冻结来源字节，不参与自动格式修改。修改它们需同步审阅来源哈希、
更新回归并重新开展真实物理资格；来源清单见
[SOURCE_PROVENANCE.json](docs/SOURCE_PROVENANCE.json)。

## 远程与多人协作

日常协作采用各自分支、提交和PR。服务器开发使用Remote-SSH打开独立checkout；
实时共同编辑使用Live Share会话。详细步骤、仓库写权限和冲突处理见
[协作说明](docs/COLLABORATION.md)与[贡献规范](CONTRIBUTING.md)。
可用Dev Containers重建CPU开发环境；仓库未配置个人服务器地址或共享密码。

## 科学状态

当前是连续时间、垂向P0表示与条件reference-water模型。新split-clock候选没有训练权重，
尚未通过完整源湖拟合、跨湖精度、温跃层、严格轨迹守恒或不确定性校准。
移包、导入及依赖元数据发生变更，因此本仓库不能继承旧运行的完整资格回执。
[科学状态](docs/SCIENTIFIC_STATUS.md)列出已实现边界与待完成事项。

原始温深、气象数组、数据库、检查点和登录凭据保留在仓库之外。
接入真实数据前应核对原划分、别名、时钟、单位、深度基准、版本与SHA；
目标内部温深只用于冻结评分，不能用于输入、初始化或模型选择。

现有[MIT许可证](LICENSE)保留；外部数据和第三方产品仍遵守各自许可。

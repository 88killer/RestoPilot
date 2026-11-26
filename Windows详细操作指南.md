# AgenticIR 当前最佳方案：Windows 详细操作指南

## 1. 这份包能做什么

本包用于复现当前“残余容忍”修复方案的运行流程：读取清晰原图 → 中心裁切 → 合成不同退化及标注 → U-Net 定位 → 排序器选择工具 → 反馈验收 → 输出修复记录及评价指标。

包中包含 seed 43 的 U-Net、当前校准排序器、Restormer、SwinIR、FBCNN、MUSIQ 所需模型权重和运行代码。**不包含清晰原始数据集，也不包含其他对照组。**

Windows 适配版已核对源码语法、模型路径和部分权重的哈希，**尚未在 Windows 实机和 GPU 上完成端到端测试**。第一次运行建议先放少量原图，确认全流程后再放完整数据。

## 2. 数据集使用什么

### 2.1 沿用上一轮 CLIC 来源

上一轮大规模实验使用 70 张清晰原图：从以下两个包各选 35 张。注意，官方网页位于 CLIC 2021 页面；其验证包文件名含 CLIC 2020，测试包文件名含 CLIC 2021。

- [CLIC 官方数据页面](https://archive.compression.cc/2021/tasks/index.html)
- [验证数据集：clic2020_professional_valid.zip](https://downloads.compression.cc/clic2020_professional_valid.zip)，约 129 MB。
- [测试数据集：clic2021_image_test.zip](https://downloads.compression.cc/clic2021_image_test.zip)，约 214 MB。

在本项目中，这些都是用来**生成退化样本的清晰参考图**；官方称作验证或测试，与本项目最终划分为验证集还是测试集是两回事。请在实验记录中保留原始来源、图像文件名、选择规则和用途。

**这个启动包不会自动选 35 + 35 张**，也不包含上一轮精确的 70 张选图清单。它会处理 `原图` 文件夹下找到的全部支持格式图片，包含子文件夹。若要逐例复现上一轮，需向项目负责人另取上一轮选图清单，并核对原图、顺序、裁切、随机种子及生成参数；单纯下载同两个数据包并不能保证得到相同的 420 例。

如果目的是最终独立测试，应先选没有参与模型训练、调参和方案选择的原图，再固定选择清单与参数。数据来源为 CLIC 并不自动证明图片或其中内容与现有模型训练集无重叠，需另外核验。

### 2.2 也可以使用其他原图

程序不要求文件名必须来自 CLIC。也可从 [DIV2K 官方页面](https://data.vision.ee.ethz.ch/cvl/DIV2K/)等来源下载清晰高分辨率图像；若研究用途是独立评估，应先核对数据划分和重叠情况。每张图的宽和高都必须至少为 512 像素。

支持 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp`。请解压数据集，把**图片文件**放到项目 `原图` 目录，不能只把数据集压缩包放进去。

### 2.3 程序怎样构造样本

按照原图文件路径排序，每张原图从中心裁切 512×512，生成以下六种情况，每种各 1 例：

| 场景 | 含义 |
|---|---|
| `compound_pair_a` | 同一区域的两种退化 |
| `compound_pair_b` | 另一组双重退化及参数组合 |
| `compound_triple` | 同一区域的噪声、模糊、JPEG 压缩三重复合退化 |
| `multi_region` | 两个区域分别包含退化 |
| `single` | 单一退化 |
| `clean` | 不添加退化的干净图 |

因此放入 70 张图会生成 **420 例，其中 210 例是同区域复合退化**。噪声、运动模糊和 JPEG 参数由脚本根据图片顺序设置；每例使用固定公式生成随机种子，并在 `manifest.json` 中记录原图文件名、原图 SHA-256、裁切坐标、退化类型及参数。改变原图数量、路径排序或顺序会改变样本编号和相应参数分配。

## 3. 硬件与软件环境

| 项目 | 本包配置或要求 |
|---|---|
| 系统 | Windows 10/11，64 位；当前适配尚未完成实机测试 |
| 显卡 | NVIDIA GPU；启动器要求 CUDA 可用。建议至少 8 GB 显存，能否稳定运行以实际显卡、驱动和可用显存为准 |
| NVIDIA 驱动 | 需要支持本包 PyTorch CUDA 12.8 运行时的驱动；用 `nvidia-smi` 检查，并在安装 PyTorch 后用 `torch.cuda.is_available()` 实测 |
| Python | **64 位 Python 3.11**；启动器调用 `py -3.11` |
| PyTorch | **2.7.1**，CUDA 12.8 构建（`cu128`） |
| torchvision | **0.22.1**，与上述 PyTorch 一起安装 |
| NumPy / SciPy | **2.3.5 / 1.16.3** |
| Pillow | **12.0.0** |
| pyiqa / lpips | **0.1.16 / 0.1.4** |
| timm / einops | **1.0.22 / 0.8.1** |
| opencv-python、PyYAML、requests | 包内 requirements 没有固定具体版本；首次安装以实际解析版本为准，并应记录 |
| 网络 | 首次需要下载 Python 依赖；模型权重已随包提供 |
| 磁盘 | 需预留 Python / PyTorch 环境、原图、生成图和逐步输出空间；逐例保存中间图，数量多时占用会增长 |

`nvidia-smi` 上的 CUDA Version 表示驱动支持的 CUDA 版本口径，**不等同于 Python 环境实际加载的 PyTorch CUDA 版本**。本包安装的是 PyTorch 官方 `cu128` 版本的二进制包，通常不需要为此单独安装完整 CUDA Toolkit；驱动仍需兼容。官方安装组合可见 [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

本包的 MUSIQ 质量评价配置为 CPU；定位与修复工具使用 CUDA GPU。整个流程耗时与显卡、图片数、工具调用次数和磁盘速度有关。

## 4. 准备与安装

### 第一步：解压工程

1. 把 `AgenticIR_Windows_最佳方案_一键启动.zip` 复制到 Windows 电脑。
2. 右键选择“全部解压”，例如解压到 `D:\AgenticIR`。
3. 进入解压后的 `AgenticIR_Windows_最佳方案` 文件夹，确认能看到：

```text
开始运行.bat
使用说明.md
Windows详细操作指南.md
requirements.txt
config.json
feedback.json
calibrated_ranker.npz
calibrated_training.json
AgenticIR\
原图\
```

建议使用有足够空余空间、可写入的本地磁盘路径。不要直接在压缩包预览窗口里双击运行。

### 第二步：安装并检查 Python 3.11

1. 从 [Python 官方 Windows 下载页面](https://www.python.org/downloads/windows/)安装 64 位 Python 3.11，并启用 Python Launcher。
2. 打开 PowerShell，执行：

```powershell
py -3.11 --version
```

看到 `Python 3.11.x` 表示启动器能找到该版本；如果提示找不到，检查 Python 3.11 与 Python Launcher 是否安装。

### 第三步：检查 NVIDIA 驱动

在 PowerShell 中执行：

```powershell
nvidia-smi
```

记录显卡型号、显存和驱动版本。如果找不到命令或没有显示 NVIDIA GPU，先检查显卡驱动。该检查通过后，仍要等 PyTorch 安装完成，再执行第 7 节的 CUDA 检查。

### 第四步：放入图片

1. 从第 2 节的官方链接下载需要的数据包，并在 Windows 上解压。
2. 选定要处理的清晰原图，复制到工程中的 `原图` 文件夹。可以放子文件夹；程序会递归查找支持格式。
3. 首次核验建议只放 1～3 张宽高均不小于 512 的原图。**程序会处理目录里的全部图片**，不要误把整个大型数据集放进去。

### 第五步：一键启动

1. 双击工程根目录的 `开始运行.bat`。
2. 首次运行会依次创建 `.venv`、安装 CUDA 版 PyTorch 和其他依赖；安装完成会写入 `.venv\setup.complete`，后续启动跳过安装。
3. 启动器检查 GPU 后，生成退化数据和清单，再执行当前修复方案。完成后窗口会提示结果位置，按任意键关闭。
4. 如果运行失败，窗口会保留报错并等待按键。请复制完整错误文字及其前面的上下文，连同 `nvidia-smi` 和第 7 节的环境检查结果发给项目负责人。

## 5. 安装包目录结构

解压后的主要目录如下；模型源码目录中的具体实现文件较多，这里按功能列出：

```text
AgenticIR_Windows_最佳方案/
├─ 开始运行.bat                     Windows 双击启动入口
├─ Windows详细操作指南.md           本指南
├─ 使用说明.md                      简明操作说明
├─ SHA256.txt                       包内文件的 SHA-256 校验清单
├─ requirements.txt                 Python 依赖及部分固定版本
├─ generate_data.py                 从清晰原图生成退化图与标注
├─ prepare_manifest.py              检查生成数据并整理实例清单
├─ run_best.py                      当前最佳方案的运行入口
├─ config.json                      定位器、评价设备和运行参数
├─ cloud_tools.json                 四个修复工具的调用模板
├─ feedback.json                    反馈验收基础配置
├─ calibrated_ranker.npz            当前校准排序器权重
├─ calibrated_training.json         排序器训练记录与权重校验值
├─ 原图/                            组员自行放入清晰原图
│  └─ 把原图放这里.txt
├─ AgenticIR/
│  ├─ __init__.py
│  ├─ region_ir/                    区域定位、决策、合成、修复和指标代码
│  │  ├─ synthesize.py              退化合成
│  │  ├─ localizer.py               U-Net 定位器加载与预测
│  │  ├─ policy.py                  工具候选与排序策略
│  │  ├─ feedback.py                修复反馈与验收
│  │  ├─ model_worker.py            修复工具推理入口
│  │  ├─ core.py                    区域修复运行流程
│  │  ├─ evaluate.py                画质指标
│  │  └─ 其他模块.py
│  ├─ runs/
│  │  └─ full_three_seed_localizers_20261029_175918/
│  │     └─ locator43/             seed 43 U-Net 权重与校准文件
│  │        ├─ best.pt
│  │        └─ calibration.json
│  └─ work/
│     ├─ models/                    修复模型所需源码
│     │  ├─ FBCNN/
│     │  ├─ Restormer/
│     │  └─ SwinIR/
│     ├─ weights/                   修复与质量评价模型权重
│     │  ├─ fbcnn_color.pth
│     │  ├─ musiq.pth
│     │  ├─ restormer_author_motion_deblurring.pt
│     │  ├─ swinir15.pth
│     │  └─ swinir50.pth
│     └─ torch_cache/hub/checkpoints/
│        └─ alexnet-owt-7be5be79.pth  LPIPS 使用的特征网络权重
└─ workspace/
   └─ outputs/持续实验_20261029/验证反馈与在线消融_20261030/
      └─ feedback.json              云端基础包中保留的原反馈配置副本
```

根目录 `feedback.json` 是当前配置引用的文件；`workspace` 中的是从基础包保留的副本。不要随意移动权重或改名，否则配置中的路径可能失效。

以下目录和文件**首次运行后才会产生**：

```text
AgenticIR_Windows_最佳方案/
├─ .venv/                           Python 虚拟环境
├─ 生成数据/                        生成的图片、标注和参数记录
├─ 实例清单.jsonl                   本批案例清单
├─ data.ready                       生成步骤完成标记
└─ 输出结果/                        修复过程、逐例指标和日志
```

## 6. 输出在哪里

```text
AgenticIR_Windows_最佳方案\
├─ 原图\                    自己放入的清晰原图
├─ 生成数据\                清晰裁切图、合成退化图、区域标注和逐例 manifest
├─ 实例清单.jsonl             所有生成案例的入口清单
├─ data.ready                生成及清单完成的标记
└─ 输出结果\
   ├─ ablation_cases.jsonl    逐例 PSNR、SSIM、LPIPS、耗时、调用及反馈记录
   ├─ ablation_config.json    本轮配置记录，成功结束后生成
   ├─ tools.json              本机工具调用路径
   └─ 过程记录\              修复过程中的图像与日志
```

虽然结果文件沿用 `ablation` 名称，本包只运行 `residual` 当前方案。指标中 PSNR、SSIM 越高通常越好，LPIPS 越低通常越好。干净参考图与输入完全相同时 PSNR 可能为无穷大，计算总体平均时应单独处理，不要直接把它当作普通有限数。

同一区域保留两步的统计单位是**区域**；零调用比例的统计单位是**样本**。耗时是该 Windows 机器的实际记录，不能与另一台机器的时间视为同硬件直接比较。

## 7. 运行后检查环境

在工程根目录打开 PowerShell。若不在根目录，先用 `cd` 进入解压后的实际路径。首次安装成功后执行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch, torchvision; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('PyTorch CUDA:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
.\.venv\Scripts\python.exe -m pip freeze > environment_versions.txt
```

期望看到 Python 3.11、PyTorch 2.7.1、torchvision 0.22.1，`torch.version.cuda` 显示 12.8，`CUDA available` 为 `True`。`environment_versions.txt` 可留作实验记录；注意其中还包含其他依赖的实际安装版本。

## 8. 换数据或重新开始

- **第一次尚未运行完成**：若 `生成数据` 已生成一部分，但没有生成 `data.ready`，请先移走或重命名不完整的 `生成数据` 再重试；脚本会避免覆盖。
- **换一批原图**：先备份或移走旧的 `生成数据`、`输出结果`、`实例清单.jsonl`、`data.ready`；再替换 `原图` 文件夹中的图片，并双击启动。
- **重跑已有生成数据**：保留 `生成数据`、`实例清单.jsonl` 和 `data.ready`，先备份或移走 `输出结果`，再启动。注意这会重新计算，不是断点续跑。
- **安装依赖中断**：再次启动会在 `setup.complete` 不存在时重试安装；若虚拟环境损坏，移走或删除项目内 `.venv` 后重新启动。
- **没有自动断点续跑**：本包会逐例写入记录，但运行中断后不应把现有结果当成已完整完成。`ablation_config.json` 是全部循环结束后写出的配置记录之一，仍需核对逐例数量和日志。

## 9. 常见问题

**提示 `py` 或 Python 3.11 找不到**：检查 Python 3.11 和 Python Launcher；用 `py -3.11 --version` 核验。

**提示 CUDA 不可用**：确认 NVIDIA 驱动正常、安装的是 `cu128` PyTorch，运行第 7 节命令并检查驱动兼容性。不要把 `nvidia-smi` 的 CUDA 显示值当成 PyTorch 已成功使用 GPU 的证明。

**提示图片尺寸小于 512**：移除或替换对应图片。不要直接拉伸图片冒充原始分辨率而不记录。

**提示文件或输出已存在**：按第 8 节先备份、移走或重命名旧数据及结果，避免覆盖。

**显存不足**：记录显卡型号、空闲显存和完整错误；不要直接更改图像尺寸、权重或模型参数后宣称与原实验完全一致。

**其他错误**：记录完整报错、`environment_versions.txt`、显卡型号和驱动版本。此 Windows 适配版尚无实机端到端通过记录，首次核验需要确认各模型和指标均能加载。

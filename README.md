# RestoPilot

**面向复杂退化图像的区域自适应修复智能体**

RestoPilot 研究如何对图像中的不同退化区域自主选择修复工具，并结合修复反馈决定是否保留结果或继续处理。项目借鉴了 [AgenticIR](https://github.com/Kaiwen-Zhu/AgenticIR) 将感知、工具调度、执行和反馈组织为修复流程的思路，在此基础上关注区域定位与局部工具调度。

> 当前仓库是研究原型。Windows 启动流程已完成静态检查，尚未经过 Windows GPU 实机端到端验证。

## 工作流程

1. **生成测试样本**：从清晰原图合成噪声、运动模糊、JPEG 压缩及其组合，同时保存区域标注。
2. **定位退化**：使用 U-Net 预测退化位置与类型，构建待修复区域。
3. **选择工具**：MLP 排序器根据区域状态、历史操作及预算，对候选修复工具打分。
4. **执行与反馈**：对选定区域调用修复模型，通过画质与区域保护等反馈规则决定是否保留结果，再根据状态决定下一步操作。
5. **记录结果**：保存 PSNR、SSIM、LPIPS、耗时、调用次数和逐步反馈等记录。

排序器使用离线采集的候选路径及收益标签训练；当前推理过程不会自动更新排序器参数。当前入口是面向**合成退化数据评估**的批量流程，需要清晰参考图来生成输入和计算参考指标。

## 环境要求

- Windows 10/11，64 位，NVIDIA GPU 与兼容 CUDA 的驱动；建议至少 8 GB 显存，实际可用性以设备为准。
- 64 位 Python 3.11 和 Python Launcher。
- 启动脚本安装 PyTorch **2.7.1**、torchvision **0.22.1** 的 **CUDA 12.8** 构建，以及 `requirements.txt` 中的依赖。
- 首次安装依赖需要联网，并为权重、运行环境、生成数据和中间结果预留磁盘空间。

Windows 版及具体依赖组合仍需在目标机器上验证。完整步骤见 [Windows 详细操作指南](Windows详细操作指南.md)。

## 下载与运行

仓库中的模型权重使用 Git LFS。先安装 [Git LFS](https://git-lfs.com/)，再在终端执行：

```bash
git lfs install
git clone https://github.com/88killer/RestoPilot.git
cd RestoPilot
```

1. 自行下载并解压清晰原始图像。可以参考 [CLIC 数据页面](https://archive.compression.cc/2021/tasks/index.html)或 [DIV2K 数据页面](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。
2. 在仓库根目录**新建** `原图` 文件夹，将要处理的图片放入其中。支持 PNG、JPG、JPEG、BMP、WEBP，宽高均需至少 512 像素；程序会递归处理该目录下的所有支持格式图片。
3. 在 Windows 上双击 `开始运行.bat`。首次运行会创建 `.venv` 并安装依赖，然后生成测试样本并运行修复。
4. 逐例指标和反馈保存在 `输出结果/ablation_cases.jsonl`，修复过程图片及日志位于 `输出结果/过程记录/`。

程序会从每张原图中心裁切 512×512，并生成两种双重退化、三重复合退化、多区域退化、单一退化和干净图六种情况。首次验证建议先使用少量图片，确认环境和输出后再运行完整数据。

重新运行前，请按[详细指南](Windows详细操作指南.md)备份或移走原来的生成数据和输出；当前流程不提供自动断点续跑。

## 目录结构

```text
RestoPilot/
├── 开始运行.bat                  Windows 启动脚本
├── Windows详细操作指南.md        数据、环境和运行说明
├── generate_data.py              合成退化数据
├── prepare_manifest.py           检查并整理样本清单
├── run_best.py                   当前残余容忍方案入口
├── config.json                   定位器及运行参数
├── feedback.json                 反馈规则配置
├── cloud_tools.json              修复工具调用配置模板
├── calibrated_ranker.npz         校准后的排序器权重
├── calibrated_training.json      排序器训练记录与权重校验
├── requirements.txt              Python 依赖
└── AgenticIR/
    ├── region_ir/                 定位、调度、反馈与评价代码
    ├── runs/                      seed 43 定位器与校准文件
    └── work/                      修复模型代码、权重和缓存
```

`AgenticIR/` 沿用了原工程目录名；当前 RestoPilot 入口使用的是区域修复流程，不是直接运行原始 AgenticIR 的完整大语言模型工作流。

## 致谢与参考

本项目受以下研究和开源实现启发，或在运行流程中使用其模型代码、权重或评价工具：

- [AgenticIR](https://github.com/Kaiwen-Zhu/AgenticIR) · [论文](https://arxiv.org/abs/2410.17809)
- [Restormer](https://github.com/swz30/Restormer)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [FBCNN](https://github.com/jiaxi-jiang/FBCNN)
- [IQA-PyTorch / pyiqa](https://github.com/chaofengc/IQA-PyTorch) 和 [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

第三方代码、模型权重及数据集分别受其原始许可约束；使用或再分发前请查阅相应项目的许可条款。

<div align="center">

![AstraLM 架构首图](assets/astralm-hero.png)

# AstraLM Lab

### 从零构建现代 Decoder-Only Transformer，并用可复现实验验证它

[在线 AstraScope](https://alex0ai.github.io/AstraLM-Lab/) · [Colab 快速体验](https://colab.research.google.com/github/Alex0AI/AstraLM-Lab/blob/main/notebooks/quickstart.ipynb) · [实验说明](docs/EXPERIMENTS.md) · [English](README.md)

</div>

AstraLM 不是只追求“能够生成文本”的小型 GPT。它重点解决三个问题：

1. KV Cache 等容易写错的部件如何进行数值级验证；
2. 架构创新如何做等参数、等 Token、多随机种子的受控实验；
3. 没有独立显卡的学习者如何通过浏览器和云端完成实验。

## 核心能力

- GQA、RoPE、RMSNorm、SwiGLU、QK-Norm、Logit Softcap；
- 分层增量 KV Cache，以及完整/缓存 Logits 一致性检查；
- 原创 RMS 平衡跨层 `Attention Bridge`；
- 完全从零实现、支持中文 UTF-8 的 Byte-Level BPE；
- 训练/验证拆分、梯度累积、混合精度、断点恢复、最佳模型保存；
- 标准残差与 Attention Bridge 的等参数多种子消融；
- 自动生成 JSON、Markdown 和独立 HTML 实验报告；
- 零依赖交互页面 AstraScope，实时估算参数量和 KV Cache。

## 最快体验

不安装任何软件，先打开 [AstraScope](https://alex0ai.github.io/AstraLM-Lab/)。

安装后运行：

```bash
pip install -e ".[dev]"
astralm estimate --preset pico
astralm verify-cache --preset pico
astralm ablate --preset pico --steps 100 --seeds 17,42,73
```

如果电脑不适合安装 PyTorch，可以在 GitHub Actions 中运行 **Cloud
TinyStories Experiment**。公开数据、训练和测试全部发生在云端 Runner，电脑
只下载最终报告。

## 项目创新边界

RoPE、GQA、RMSNorm、SwiGLU、QK-Norm 等是公开的成熟技术，不属于本项目
原创。AstraLM 的实验性创新是 Attention Bridge，以及围绕它设计的等参数
验证体系。项目不会在缺少数据时宣称它普遍优于标准 Transformer。

详细来源与致谢见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。

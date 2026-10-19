# KV 页粒度项目：报告与中文展示快照（2026-10-19）

本目录保留项目文档、图表和制作脚本当时的相对路径，独立于仓库根目录的实验展示版本。数值与实验执行代码相同，图表排版可能不同。所有以下命令均从**本目录**运行。

- [一页项目规格 PDF](output/pdf/proposal.pdf) 与 [LaTeX 源码](course/project/proposal.tex)：原型完成后整理，未作为过去的提案提交。
- [英文报告 PDF](output/pdf/report.pdf) 与 [LaTeX 源码](course/project/report.tex)：正文 5 页、参考文献 1 页，共 6 页；采用带自学声明的官方 MLSys 2025 模板，不代表会议接收。
- [中文展示 MP4](output/video/kv-cache-presentation.mp4)：540.14 秒、1280×720、12 章。ZIP 不重复打包 MP4，可从[博客下载视频](https://liangqianxing.github.io/materials/cmu-15-779/project/kv-cache-presentation-20261019.mp4)。
- [中文讲稿](course/project/presentation_transcript.md)、[视频时长与哈希](output/video/presentation_manifest.json)、[依赖和制作脚本](course/project/make_video.py)。视频为 macOS Tingting 合成语音，非学习者录音。
- [原始 210 行结果](results/project_kv_granularity.json)、[汇总](course/project/summary.json)、[图表](course/project/figures/)、[文件哈希清单](manifest.json)。

报告、代码和展示由 Codex 辅助制作。归档证明材料已经生成并经过文件/版式/媒体检查；不代表本人掌握、课堂参与、正式课程提交或整门课程完成。实验仍为 CPU 合成单头 K/V/Q，未测完整 LLM、GPU KV kernel、多卡、真实服务吞吐或进程峰值内存。

## 实验与图表

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r course/project/requirements.txt
.venv/bin/python course/project/run_experiments.py --output results/project-rerun.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python course/project/analyze_results.py
```

分析脚本默认读取归档的原始 JSON，不会自动切换到另存的复跑结果。上述两项项目测试检查策略数值一致和页边界/复制计数；随机页池测试等完整 54 项测试见外层仓库 `tests/`。完整扫描已在记录 14 的实验发布时复验，此次文档归档没有重新计时，也不替换原始计时。

## PDF

安装 Tectonic 后，从本目录执行：

```bash
mkdir -p output/pdf
cd course/project
tectonic -X compile -Z search-path=template -o ../../output/pdf report.tex
tectonic -X compile -o ../../output/pdf proposal.tex
```

模板来源和原始 ZIP 哈希见 `course/project/template/source.json`。随附 PDF 保留源聊天生成的原始字节；本次逐页检查了 1 页规格和 6 页报告。不同 TeX 版本重编译不保证相同 PDF 字节。

## 中文视频

回到本目录运行 `.venv/bin/python course/project/make_video.py`。需要 macOS `say -v Tingting`、`/System/Library/Fonts/Supplemental/Arial Unicode.ttf`、Pillow 和 imageio-ffmpeg；不适用于未经调整的 Linux 环境。脚本会在本目录的 `build/project-video/` 写中间文件，并覆盖讲稿、视频与视频清单，复做时建议复制目录。

本次核对 12 章画面及讲稿与制作脚本的一致性，并完整解码 MP4；没有把媒体解码当作学习者口头展示或理解能力的验收。视频中的可能扩展仅为讨论。

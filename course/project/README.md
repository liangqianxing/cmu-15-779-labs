# KV cache 页粒度实验

发布范围是已经运行的 CPU 系统实验、原始数据、图表生成和两项现有项目测试。PDF、演示视频和正式课程提交仍独立验收，本目录不作项目整体完成声明。

```bash
python -m pip install -r course/project/requirements.txt
python course/project/run_experiments.py --output results/project-rerun.json
python course/project/analyze_results.py
python -m unittest discover -s tests -v
```

分析脚本默认读取 results/project_kv_granularity.json，重新生成本目录的图表、汇总与表格；不会自动分析上面另存的复跑文件。请保留原始报告，明确选择分析版本。

发布版只调整了绘图排版：存储图采用两行面板，关闭重叠的对数次刻度，将计时图图例移到曲线外。原始 JSON、实验执行脚本和数值汇总未改变。

共5类合成工作负载、3个种子、静态/动态基线及6种块长的独立/共享分页，210组配置、1470次原始计时。每请求固定K/V/Q，非完整LLM。active_slot_bytes、pool_tensor_bytes、copied_old_token_slots分列，未测量RSS或进程峰值内存，CPU分页路径慢于连续存储基线。

[学习记录14](https://liangqianxing.github.io/posts/cmu-15-779-kv-cache-granularity-study/)解释前缀对齐、写时复制及计时边界。sources中的SHA-256可核对运行代码与33号缓存；历史gpu_status字段反映当时CPU项目安排，不否认后续独立GPU实验。

# HQA-IQL 项目说明

本项目将原始源码与已完成实验中的数据处理、算法对照、并行训练和结果核验整理为统一工程，包含 **60 个训练任务的配置与真实结果摘要**。研究按“方法对照”和“组件消融”组织，实验顺序不作为项目结构。

[English README](README.md) · [算法细节](docs/METHOD.md) · [数据说明](docs/DATA.md) · [运行协议](docs/EXPERIMENTS.md) · [完整结果表](results/README.md) · [本次验证范围](docs/VALIDATION.md)

## 上传 GitHub

将本目录中的文件上传为仓库根目录即可，保留 `.github`、`.gitignore` 和 `.gitattributes`。压缩包用于传输，上传仓库时应上传解压后的源码文件。

项目保留源码、60 份配置、数据校验清单、测试、文档和小体积结果 CSV。训练数据、模型权重、原始日志归档、虚拟环境及个人论文材料不在发布包中；本地原文件不受影响。后续生成的数据和训练输出已配置为 Git 忽略项。

## AutoDL 安装

将工程解压到实例中，例如 `~/autodl-tmp/HQA-IQL-GitHub`，使用 Python 3.10–3.12：

```bash
cd ~/autodl-tmp/HQA-IQL-GitHub
bash autodl/setup.sh
source autodl/env.sh
```

默认安装 PyTorch 2.7.1 的 CUDA 11.8 构建，其余库版本固定在 `requirements.txt`。需要其他构建时，可使用 `TORCH_CUDA=cu128 bash autodl/setup.sh`，也支持 `cu126`；不要把 `nvidia-smi` 显示的 CUDA 版本直接当成已安装的 PyTorch 版本。每次重新打开终端后，执行 `source autodl/env.sh`。

## 导入已有数据

直接复用已保存的实验项目包，无需重新下载：

```bash
python scripts/import_data.py --archive /你的文件路径/HQA_IQL_AutoDL_Supplement_20261017.zip
python scripts/preflight.py
```

也可以使用 `--source-dir /已解压的数据目录`。导入工具只读取指定的数据文件，并逐一核对 SHA-256，不会执行归档中的代码。文件将放到 `data/processed/` 和 `minari_datasets/`。如果你已设置 `MINARI_DATASETS_PATH`，运行时会使用该路径，因此应保证其中的数据与校验清单一致。

预检查会核对数据、权重变体、60 份配置、GPU 运算，以及实际 Kitchen 环境的观测、奖励与 280 步时限。只做 CPU 诊断时可加 `--allow-cpu`；正式训练和烟雾测试需要 CUDA。原始 Kitchen 数据的重建方式及限制见[数据说明](docs/DATA.md)。

## 并行运行与续跑

查看全部任务，不加载训练数据：

```bash
python scripts/run_experiments.py --preset all --dry-run
```

短流程检查：

```bash
python scripts/run_experiments.py --preset comparisons --methods baseline full --datasets mixture --smoke --jobs 2 --threads 1
```

完整复现实验：

```bash
python scripts/run_experiments.py --preset all --jobs 6 --threads 1
```

按需要筛选任务，例如只运行混合数据上的 HQA-IQL 和 IQL-Ref：

```bash
python scripts/run_experiments.py --preset comparisons --methods full reference_iql --datasets mixture --seeds 3 4 5 --jobs 3
python scripts/status.py --preset all
python scripts/collect_results.py
```

`--jobs` 是同时运行的训练进程数，`--threads` 是每个进程的 CPU 线程数。锁只约束同一个任务，不同方法、不同 seed 可以在同一目录并行。并发数应结合显存、内存和 CPU 负载调整。

中断后重复原命令即可续跑；已完成任务会核对输出后跳过。为了避免混用实验，代码、配置或数据改变后，旧任务不会直接续跑，应保留旧工程并在另一份工程中开展新实验。训练输出位于 `runs/`，短流程检查位于独立的 `runs/smoke/`，不会混入正式结果。

## 实验与结果口径

| 实验组 | 模型数 | 训练 seed | 验证环境 seed | 测试环境 seed |
|---|---:|---|---|---|
| 方法对照 `comparisons` | 42 | 3、4、5 | 11000–11019 | 30000–30049 |
| 组件消融 `ablations` | 18 | 0、1、2 | 10000–10019 | 20000–20049 |

每个模型训练 200,000 次更新，每 5,000 次更新验证 20 个 episode。以验证回报首次达到严格最大值的检查点作为选中模型，同时报告最终训练检查点；二者各测试 50 个 episode。统计单位为训练 seed，报告三个 seed 的均值与样本标准差，两组实验不能合并为六个 seed。

当前结果显示 HQA-IQL 的收益依赖数据条件，不能概括为全面优于各基线。训练末段性能下降明显，因此结果表同时保留选中模型和训练终点。任务采用统一的四任务 Kitchen 观测与持续奖励，回报为原始回报，不是 D4RL 标准化分数。消融组的测试 seed 20000、20001 曾用于工程检查，这一事实保留在实验协议中。

`results/recorded_runs.csv` 包含 120 条模型检查点评估记录，`results/validation_curves.csv` 包含 2,400 条验证记录，`results/summary.csv` 包含 40 组统计。重新计算统计表不需要训练或安装额外依赖：

```bash
python scripts/summarize_recorded_results.py
```

本次整理没有重新执行完整训练。已完成实验的结果作为记录提供；新工程的代码检查与短流程验证不等同于重新得到相同实验分数。

## 开发检查与许可

安装 PyTorch 后，可执行：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest
```

测试使用合成数据，不要求下载研究数据。GitHub Actions 已配置 CPU 测试和任务规划检查。

上游 IQL 文件保留原 MIT 许可及来源，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原项目代码暂未指定仓库整体的开源许可；这与上游文件许可分别处理。

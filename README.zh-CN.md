# De-NATRM

`De-NATRM` 是与论文配套发布的正式代码仓库，对应 HER2 阳性乳腺癌新辅助治疗深度因果推荐模型。仓库提供主训练链路、最终冻结模型包，以及面向正式公开发布的验证材料。

## 发布快照

- 状态：论文配套正式代码发布版
- 版本：`v1.0.0`
- 许可证：`MIT`
- 主入口：`run.py`
- 冻结模型包：`export/model_package/`

该公开仓库仅保留 De-NATRM 主模型发布所需内容，不分发完整私有研究工作区中的探索性实验历史、受保护的临床表格、论文制图工作目录或辅助分析脚本。

## 仓库范围

这个公开仓库主要服务于：

- 代码检查与方法理解
- 与论文配套的复现边界说明
- 冻结模型包推理验证
- 发布前仓库验证

受保护的临床源数据不会随本仓库公开发布，因此完整训练复现需要用户自行提供经过授权的数据资产。

## 核心代码主链路

De-NATRM 的主训练链路为：

```text
config.py -> data.py -> model.py -> train.py -> run.py
```

- `config.py` 负责运行配置以及公开发布后的路径覆盖入口
- `data.py` 负责授权数据加载、预处理、特征组装和元数据导出
- `model.py` 实现 De-NATRM 的核心架构 `CausalNet`
- `train.py` 实现事实损失、MMD 和反事实一致性训练
- `run.py` 是默认的端到端训练与评估入口
- `export/model_package/` 保存最终冻结模型包及其验证脚本

## 安装

安装公开版本的主要依赖：

```bash
pip install -r requirements.txt
```

## 数据可用性

本仓库不会公开完整训练所需的临床源表、固定划分表和患者级派生输出。

在尝试训练前，请先阅读 [docs/github-release/data_availability.md](docs/github-release/data_availability.md)。

如果你拥有经过授权的本地数据，可以通过环境变量配置训练入口：

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=0 \
DE_NATRM_DATA_PATH=/path/to/authorized_dataset.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

如果使用固定划分模式，也支持：

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=1 \
DE_NATRM_TRAIN_DATA_PATHS=/path/train_batch1.xlsx:/path/train_batch2.xlsx:/path/train_batch3.xlsx \
DE_NATRM_TEST_DATA_PATH=/path/test.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

## 验证

运行公开发布相关的验证：

```bash
python -m pytest tests/test_model_package.py tests/test_release_docs.py
python export/model_package/scripts/validate_package.py --package-dir export/model_package
```

运行冻结模型包中的单例推理示例：

```bash
python export/model_package/scripts/predict_package.py \
  --input export/model_package/examples/example_patient_input.json
```

## 发布文档

- [核心代码导览](docs/core_code_map.md)
- [数据可用性说明](docs/github-release/data_availability.md)
- [发布前检查清单](docs/github-release/release_checklist.md)
- [模型资产审计说明](docs/model_asset_audit.md)
- [最终模型包说明](export/model_package/README.md)

## 引用

如果你在研究中使用了本仓库，请同时引用论文和 [CITATION.cff](CITATION.cff) 中的仓库元数据。

## 许可证

本仓库采用 [MIT License](LICENSE)。

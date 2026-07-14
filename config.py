import os


def _env_text(name: str):
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _env_path(name: str, default: str) -> str:
    value = _env_text(name)
    if value is None:
        return default
    return _normalize_path(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _env_path_list(name: str, default: list[str]) -> list[str]:
    value = _env_text(name)
    if value is None:
        return default
    separator = os.pathsep if os.pathsep in value else ","
    items = [_normalize_path(part.strip()) for part in value.split(separator) if part.strip()]
    return items or default

# === 数据路径配置 ===
# 公开仓库默认不携带受保护训练数据。
# 如需运行训练，请通过环境变量提供经过授权的本地路径。
BASE_DIR = os.path.dirname(__file__)
_DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "authorized_dataset.xlsx")
DATA_PATH = _env_path("DE_NATRM_DATA_PATH", _DEFAULT_DATA_PATH)
USE_FIXED_SPLIT_FILES = _env_bool("DE_NATRM_USE_FIXED_SPLIT_FILES", False)
_DEFAULT_TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data_splits", "authorized", "train_batch1.xlsx")
TRAIN_DATA_PATH = _env_path("DE_NATRM_TRAIN_DATA_PATH", _DEFAULT_TRAIN_DATA_PATH)
_DEFAULT_TRAIN_DATA_PATHS = [
    os.path.join(BASE_DIR, "data_splits", "authorized", "train_batch1.xlsx"),
    os.path.join(BASE_DIR, "data_splits", "authorized", "train_batch2.xlsx"),
    os.path.join(BASE_DIR, "data_splits", "authorized", "train_batch3.xlsx"),
]
TRAIN_DATA_PATHS = _env_path_list("DE_NATRM_TRAIN_DATA_PATHS", _DEFAULT_TRAIN_DATA_PATHS)
_DEFAULT_TEST_DATA_PATH = os.path.join(BASE_DIR, "data_splits", "authorized", "test.xlsx")
TEST_DATA_PATH = _env_path("DE_NATRM_TEST_DATA_PATH", _DEFAULT_TEST_DATA_PATH)
_DEFAULT_VARPOOL_PATH = os.path.join(BASE_DIR, "data", "authorized_varpool.xlsx")
_varpool_candidate = _env_path("DE_NATRM_VARPOOL_PATH", _DEFAULT_VARPOOL_PATH)
VARPOOL_PATH = _varpool_candidate if os.path.exists(_varpool_candidate) else None
# 公开仓库默认输出目录
OUTPUT_DIR = _env_path("DE_NATRM_OUTPUT_DIR", os.path.abspath(os.path.join(BASE_DIR, "outputs", "de_natrm_run")))
CODE_VERSION = "de_natrm_public_release_snapshot_20260316"

# === 关键列配置 ===

EXPLICIT_OUTCOME_COL = "tpCR"      # 结果列名
EXPLICIT_TREATMENT_COL = "Treatment"    # 治疗列名

#EXPLICIT_OUTCOME_COL = "原发灶疗效"      # 结果列名
#EXPLICIT_TREATMENT_COL = "NAT方案性质"    # 治疗列名
MP_COL_CANDIDATES = ["MP", "MP_grade", "MP分级", "Miller-Payne", "MillerPayne"]
def MP_AS_PCR_RULE(s): 
    # MP分级5为pCR
    try:
        return (s.astype(str).str.strip() == "5") | (s.astype(float) == 5)
    except Exception:
        return s.astype(str).str.strip().eq("5")

# === 特征选择配置 ===
# 是否在建模中使用 C2 后随访数据（如 “C2后Size-MR / C2后SUVmax”）
# False为只用基线特征；后续若要加入，只需改为 True 并重新训练模型。
USE_C2_FEATURES = False

# 可选：显式指定仅保留的表格特征白名单。
# None 或空列表表示不启用，仍按变量池/自动筛列逻辑选择。
TABULAR_FEATURE_WHITELIST = None

# 是否接入导出的 MR/PET ViT embedding 作为独立高维分支
# 当前默认关闭，改为优先生成少量图像风险分数后回灌到主模型
USE_IMAGE_EMBEDDINGS = False
IMAGE_EMBEDDING_DIR = os.path.join(BASE_DIR, "derived_assets", "authorized", "embedding_exports")
# 可选: "mr", "pet", "mr,pet"
IMAGE_EMBED_MODALITIES = "mr,pet"
# token 聚合方式: "mean" 或 "meanstd"
IMAGE_EMBED_POOLING = "mean"
# 训练集内 PCA 降维后的目标维度
IMAGE_EMBED_MR_DIM = 8
IMAGE_EMBED_PET_DIM = 16
# 是否在主干额外加入 embedding 缺失指示位
IMAGE_EMBED_ADD_MISSING_FLAGS = True
# 是否将图像 embedding 先压缩成少量风险分数，再作为普通数值特征接回主模型
USE_IMAGE_RISK_SCORES = True
# image risk score 生成模式:
# - "global": 全体样本学一个统一 pCR 风险分数
# - "treatment_aware": 按治疗方案分别学图像风险分数
IMAGE_RISK_SCORE_MODE = "global"
# image risk score 底层打分器:
# - "logistic": PCA 后 embedding -> LogisticRegression
# - "mlp": PCA 后 embedding -> 小型 MLPClassifier
IMAGE_RISK_MODEL_TYPE = "logistic"
# 当 IMAGE_RISK_MODEL_TYPE="mlp" 时生效
IMAGE_RISK_MLP_HIDDEN = (16,)
IMAGE_RISK_MLP_ALPHA = 1e-2
IMAGE_RISK_MLP_MAX_ITER = 1000
# 是否将 image risk score 相关特征从主干表格中拆出，走独立 image branch
IMAGE_RISK_USE_SEPARATE_BRANCH = False
# 公开仓库当前仅保留原始 image risk score 分支模式。
IMAGE_RISK_BRANCH_FEATURE_MODE = "raw"
# 兼容旧逻辑：保留 USE_PET_FEATURES 名称，实际复用统一 image embedding 开关
USE_PET_FEATURES = USE_IMAGE_EMBEDDINGS

# ROI 开关：
# - "off": 不使用 ROI
# - "on": 使用 ROI 特征
# - "strict": 使用 ROI 特征，并仅保留 ROI 完整且无多序列歧义的病例
# 平时只改这一项即可。
ROI_MODE = "on"
ROI_FEATURE_TABLE_PATH = os.path.join(BASE_DIR, "derived_assets", "authorized", "roi_features", "roi_features_pyradiomics_v3_live.csv")
# 如需切换到更标准的 radiomics ROI 表，可改成：
# ROI_FEATURE_TABLE_PATH = os.path.join(BASE_DIR, "analysis", "roi_features", "roi_features_radiomics.csv")
# 如需切换到官方 PyRadiomics 版，可改成：
# ROI_FEATURE_TABLE_PATH = os.path.join(BASE_DIR, "analysis", "roi_features", "roi_features_pyradiomics.csv")

# ROI radiomics 筛选：
# - "none": 不做额外筛选
# - "train_radiomics": 仅在训练集内对 ROI 扩展特征做缺失/相关/单变量筛选
# - "train_radiomics_residual": 先拟合 no-ROI baseline，再按单个 ROI 对 baseline logloss 的交叉验证增益打分
ROI_FEATURE_SELECTION_MODE = "train_radiomics_residual"
ROI_RAD_MAX_MISSING = 0.35
ROI_RAD_CORR_THRESHOLD = 0.95
# True 时，相关性去冗余优先只在同模态内进行，避免 MR 被 PET 的高分特征整体挤掉
# 当前默认保持旧最优策略；需要做更均衡的 ROI 实验时再打开。
ROI_RAD_CORR_WITHIN_MODALITY_ONLY = False
ROI_RAD_MAX_FEATURES = 4
ROI_RAD_MAX_PET_FEATURES = 0
# ROI radiomics 消融开关：
# - ROI_RAD_MODALITIES: "all", "mr", "pet"
# - ROI_RAD_CLASSES: "all" 或逗号分隔的 radiomics 类别
#   支持: shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm
ROI_RAD_MODALITIES = "pet"
ROI_RAD_CLASSES = "shape"
# 若为 True，且 modalities="all"，训练集内 radiomics 筛选会尽量按 MR/PET 均衡分配名额
ROI_RAD_BALANCE_MODALITIES = False
# 可选：在总 radiomics 名额内，额外给 texture 类保留少量预算
# 例如：
# ROI_RAD_CLASSES = "shape,firstorder,glcm,glszm"
# ROI_RAD_TEXTURE_CLASSES = "glcm,glszm"
# ROI_RAD_TEXTURE_BUDGET = 6
ROI_RAD_TEXTURE_CLASSES = ""
ROI_RAD_TEXTURE_BUDGET = 0
# ROI radiomics harmonization:
# 仅在固定 train/test 模式下，对训练集内来自不同训练批次的扩展 ROI radiomics 做分布对齐
ROI_RAD_HARMONIZE = True
ROI_RAD_HARMONIZE_MIN_BATCH_N = 20
# ROI 稳定性筛选（训练批次间 max|SMD| 过滤）：
# >0 时仅保留 max|SMD| <= 该阈值的 ROI 扩展特征；<=0 关闭
ROI_RAD_MAX_BATCH_SMD = 0.25
# 可选：为不同模态设置独立稳定性阈值；>0 时优先覆盖全局阈值
# 当前默认保持关闭，以维持既有最优配置；做 ROI 平衡实验时可单独打开。
ROI_RAD_MAX_BATCH_SMD_MR = 0.35
ROI_RAD_MAX_BATCH_SMD_PET = 0.25
ROI_RAD_STABILITY_MIN_BATCH_N = 20
# 是否追加一小组固定的 MR/PET ROI 交互特征
ROI_ADD_INTERACTIONS = False
ROI_ADD_PRIMARY_COMPOSITES = True
# 是否将 ROI 数值特征从主干中拆出，走独立 ROI branch
ROI_USE_SEPARATE_BRANCH = True
# ROI branch 对缺失/歧义 ROI 的处理：
# - "impute": 仍按数值列统一插补
# - "gate": 保留样本，但将该样本 ROI branch 置零，只用表格主干
ROI_BRANCH_MISSING_POLICY = "gate"

# 若用 run_experiment_from_config.py，可选是否自动给输出目录追加 ROI 模式后缀
AUTO_APPEND_ROI_MODE_TO_OUTPUT_DIR = False

# 可选：给输出目录再追加一个自定义 tag，便于调参时区分实验
# 例如 "lr5e4_seed42"
EXPERIMENT_TAG = ""

_ROI_MODE_NORMALIZED = str(ROI_MODE).strip().lower()
if _ROI_MODE_NORMALIZED not in {"off", "on", "strict"}:
    raise ValueError(f"Unsupported ROI_MODE: {ROI_MODE!r}. Expected one of: off, on, strict.")

# 兼容现有训练入口：真正被 data.py 使用的仍是这两个布尔开关
USE_ROI_FEATURES = _ROI_MODE_NORMALIZED in {"on", "strict"}
ROI_STRICT_SENSITIVITY = _ROI_MODE_NORMALIZED == "strict"

# === 超参数 ===
TOP_K_TREATMENTS = 2   # 治疗数量K
RANDOM_STATE = 42       # 随机种子
BATCH_SIZE = 16         # 批次大小
EPOCHS = 170            # 训练轮数
#LR = 1.1e-3            # 原学习率
LR = 1.45e-4           # 学习率（smallhead_lr145_harm_on）
#WEIGHT_DECAY = 1e-4     # 原权重衰减
WEIGHT_DECAY = 0.024    # 权重衰减



# === 因果正则化参数  ===
#ALPHA_MMD = 0.5    # MMD
#BETA_CF = 0.3     # 反事实损失
ALPHA_MMD = 0.10     # MMD（smallhead_lr145_harm_on）
BETA_CF = 0.015      # 反事实损失（smallhead_lr145_harm_on）
MMD_SIGMA = 0.5        # MMD核函数参数

# === 反事实匹配参数 ===
CF_METHOD = 'knn'      # 匹配方法: 'random', 'knn', 'knn_same_outcome'
CF_K = 5               # 近邻数K
CF_DISTANCE = 'cosine' # 距离度量: 'euclidean', 'cosine'
CF_LABEL_MODE = 'observed'   # 'observed', 'predicted', 'blended'
CF_BLEND_LAMBDA = 0.2        # blended 模式下预测伪标签占比
CF_DISTANCE_WEIGHTED = True  # 是否按匹配距离加权邻居标签
CF_LABEL_SMOOTHING = 0.02    # 软标签平滑，避免0/1极值
CF_WARMUP_EPOCHS = 50         # 前多少个epoch不启用CF loss
CF_RAMP_EPOCHS = 20           # warmup后用多少个epoch把beta线性拉到目标值

# ===  网络结构  ===
REP_DIM = 24           # 表示维度
ENC_HIDDEN = [48, 24]  # 编码器隐藏层
HEAD_HIDDEN = [8]      # 小 head
DROPOUT = 0.46         # Dropout
BN = True              # 启用BN 

# === 结果头损失配置 ===
# - "bce": 标准 BCEWithLogitsLoss
# - "weighted_bce": 对正负样本做加权 BCE
# - "focal": focal BCE，用于强调 hard examples
FACTUAL_LOSS_TYPE = "bce"
FACTUAL_POS_WEIGHT = 1.0
FACTUAL_NEG_WEIGHT = 1.0
FACTUAL_FOCAL_GAMMA = 2.0
# focal alpha 代表正类权重；当正类占多数且想抑制假阳性时，可设 <0.5
FACTUAL_FOCAL_ALPHA = 0.5

# === AUROC 置信区间（bootstrap，可选）===
# True：对 train/test 的 AUROC 做有放回 bootstrap，输出 95% CI
BOOTSTRAP_AUROC_CI = False
BOOTSTRAP_N = 1000
BOOTSTRAP_CI_LEVEL = 0.95

# === 阈值优化策略 ===

THRESHOLD_STRATEGY = 'fixed'
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.80

# === 训练集内验证与早停（不改变最终测试集） ===
ENABLE_VAL_EARLY_STOPPING = False
VAL_SPLIT_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 12

# === 训练采样重加权（应对批次分布偏移） ===
USE_BATCH_BALANCED_SAMPLER = False
BATCH_SAMPLER_GROUP_BY_Y = False
BATCH_SAMPLER_POWER = 0.7

# 批量调参时可关闭训练后自动嵌入可视化，加速实验
AUTO_AFTER_TRAINING_VIS = False

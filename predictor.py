"""
预测模块 - 负责模型预测和结果生成

该模块包含所有与预测相关的功能：
- 生成预测结果表
- 创建热力图矩阵
- 生成推荐报告
- 处理全数据预测
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from config import DATA_PATH, OUTPUT_DIR


class PredictionTableGenerator:
    """预测结果表生成器"""
    
    def __init__(self, keep: List[str], t_map: Dict[str, int]):
        """
        初始化预测表生成器
        
        Args:
            keep: 治疗方案名称列表
            t_map: 治疗方案名称到索引的映射
        """
        self.keep = keep
        self.t_map = t_map
        self.df_original = pd.read_excel(DATA_PATH, engine="openpyxl")
    
    def create_prediction_table(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray, 
                              t: np.ndarray, prefix: str = "", train_size: int = None) -> Dict[str, Any]:
        """
        创建预测结果表的通用函数
        
        Args:
            X: 特征矩阵
            y: 真实标签
            probs: 预测概率矩阵 (n_samples, n_treatments)
            t: 治疗方案索引
            prefix: 数据集前缀 ("train", "test", "all")
            train_size: 训练集大小（用于确定测试集索引）
            
        Returns:
            包含预测结果的字典
        """
        # 1. 基础信息
        result_data = {
            "患者ID": range(1, len(probs) + 1),
            "实际pCR结果": y,
        }
        
        # 2. 添加影像ID和实际治疗方案
        image_ids, actual_treatment_names = self._get_metadata(prefix, len(probs), train_size, t)
        result_data["影像ID"] = image_ids
        result_data["实际治疗方案"] = actual_treatment_names
        
        # 3. 计算实际治疗方案的预测pCR率
        actual_treatment_probs = self._calculate_actual_treatment_probs(actual_treatment_names, probs)
        result_data["实际方案预测pCR率"] = actual_treatment_probs
        
        # 4. 计算推荐治疗方案和提升空间
        best_treatment_names, best_probs, uplift = self._calculate_recommendations(probs)
        result_data["推荐治疗方案"] = best_treatment_names
        result_data["推荐方案预测pCR率"] = best_probs
        result_data["提升空间"] = uplift
        
        # 5. 添加所有治疗方案的预测pCR率
        self._add_all_treatment_probs(result_data, probs)
        
        return result_data
    
    def _get_metadata(self, prefix: str, n_samples: int, train_size: int, t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """获取影像ID和实际治疗方案名称"""
        if prefix == "train":
            # 训练集：使用训练集的实际治疗方案名称
            actual_treatment_names = [self.keep[t_val] for t_val in t]
            # 找到训练集样本在原始数据中的索引
            train_indices = np.arange(n_samples)
            image_ids = self.df_original.iloc[train_indices]["影像ID"].values
        elif prefix == "test":
            # 测试集：使用测试集的实际治疗方案名称
            actual_treatment_names = [self.keep[t_val] for t_val in t]
            # 找到测试集样本在原始数据中的索引
            if train_size is None:
                train_size = n_samples
            test_indices = np.arange(train_size, train_size + n_samples)
            image_ids = self.df_original.iloc[test_indices]["影像ID"].values
        else:  # all data
            # 全数据：使用全数据的实际治疗方案名称
            actual_treatment_names = [self.keep[t_val] for t_val in t]
            image_ids = self.df_original["影像ID"].values
        
        return image_ids, actual_treatment_names
    
    def _calculate_actual_treatment_probs(self, actual_treatment_names: List[str], probs: np.ndarray) -> List[float]:
        """计算实际治疗方案的预测pCR率"""
        actual_treatment_probs = []
        for i, actual_treatment in enumerate(actual_treatment_names):
            if actual_treatment in self.t_map:
                treatment_idx = self.t_map[actual_treatment]
                actual_treatment_probs.append(probs[i, treatment_idx])
            else:
                actual_treatment_probs.append(0.0)
        return actual_treatment_probs
    
    def _calculate_recommendations(self, probs: np.ndarray) -> Tuple[List[str], List[float], List[float]]:
        """计算推荐治疗方案和提升空间"""
        best_treatment_names = []
        best_probs = []
        uplift = []
        
        for i in range(len(probs)):
            # 找到预测pCR率最高的治疗方案
            best_idx = np.argmax(probs[i])
            best_treatment = self.keep[best_idx]
            best_prob = probs[i, best_idx]
            
            best_treatment_names.append(best_treatment)
            best_probs.append(best_prob)
            
            # 计算提升空间（最高预测pCR率 - 实际治疗方案预测pCR率）
            actual_prob = probs[i, 0]  # 假设第一个是实际治疗方案
            uplift.append(best_prob - actual_prob)
        
        return best_treatment_names, best_probs, uplift
    
    def _add_all_treatment_probs(self, result_data: Dict[str, Any], probs: np.ndarray):
        """添加所有治疗方案的预测pCR率"""
        for j, treatment_name in enumerate(self.keep):
            col_name = f"pCR率[{treatment_name}]"
            result_data[col_name] = probs[:, j].tolist()


class HeatmapGenerator:
    """热力图矩阵生成器"""
    
    def __init__(self, keep: List[str]):
        """
        初始化热力图生成器
        
        Args:
            keep: 治疗方案名称列表
        """
        self.keep = keep
    
    def generate_heatmap_matrix(self, df: pd.DataFrame, prefix: str) -> str:
        """
        生成热力图矩阵
        
        Args:
            df: 包含预测结果的DataFrame
            prefix: 数据集前缀 ("train", "test")
            
        Returns:
            保存的文件路径
        """
        # 提取治疗方案列
        treatment_cols = [col for col in df.columns if col.startswith('pCR率[')]
        treatment_names = [col.replace('pCR率[', '').replace(']', '') for col in treatment_cols]
        
        # 创建热力图矩阵
        heatmap_data = {}
        for i, col in enumerate(treatment_cols):
            heatmap_data[treatment_names[i]] = df[col].values
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.index = df['患者ID']
        
        # 保存热力图矩阵
        matrix_path = os.path.join(OUTPUT_DIR, f"{prefix}_heatmap_matrix.csv")
        heatmap_df.to_csv(matrix_path, encoding='utf-8-sig')
        
        return matrix_path


class RecommendationGenerator:
    """推荐报告生成器"""
    
    def __init__(self, keep: List[str]):
        """
        初始化推荐生成器
        
        Args:
            keep: 治疗方案名称列表
        """
        self.keep = keep
    
    def generate_recommendations(self, result_data: Dict[str, Any], prefix: str) -> str:
        """
        生成推荐报告
        
        Args:
            result_data: 预测结果数据
            prefix: 数据集前缀 ("train", "test")
            
        Returns:
            保存的文件路径
        """
        # 创建推荐报告
        rec_data = {
            "患者ID": result_data["患者ID"],
            "影像ID": result_data["影像ID"],
            "实际治疗方案": result_data["实际治疗方案"],
            "实际方案预测pCR率": result_data["实际方案预测pCR率"],
            "推荐治疗方案": result_data["推荐治疗方案"],
            "推荐方案预测pCR率": result_data["推荐方案预测pCR率"],
            "提升空间": result_data["提升空间"]
        }
        
        rec_df = pd.DataFrame(rec_data)
        
        # 保存推荐报告
        rec_path = os.path.join(OUTPUT_DIR, f"{prefix}_recommendations.csv")
        rec_df.to_csv(rec_path, index=False, encoding='utf-8-sig')
        
        return rec_path


class PredictionPipeline:
    """预测流水线 - 整合所有预测相关功能"""
    
    def __init__(self, keep: List[str], t_map: Dict[str, int]):
        """
        初始化预测流水线
        
        Args:
            keep: 治疗方案名称列表
            t_map: 治疗方案名称到索引的映射
        """
        self.table_generator = PredictionTableGenerator(keep, t_map)
        self.heatmap_generator = HeatmapGenerator(keep)
        self.recommendation_generator = RecommendationGenerator(keep)
        self.keep = keep
        self.t_map = t_map
    
    def process_predictions(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray, 
                          t: np.ndarray, prefix: str, train_size: int = None) -> Dict[str, str]:
        """
        处理预测结果，生成所有相关文件
        
        Args:
            X: 特征矩阵
            y: 真实标签
            probs: 预测概率矩阵
            t: 治疗方案索引
            prefix: 数据集前缀
            train_size: 训练集大小
            
        Returns:
            包含生成文件路径的字典
        """
        # 生成详细预测结果表
        result_data = self.table_generator.create_prediction_table(X, y, probs, t, prefix, train_size)
        detailed_df = pd.DataFrame(result_data)
        
        detailed_path = os.path.join(OUTPUT_DIR, f"{prefix}_per_treatment_probs_detailed.csv")
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        print(f"[INFO] {prefix}详细预测结果已保存到: {detailed_path}")
        
        # 生成简化预测结果表
        simple_df = self._create_simplified_table(result_data)
        simple_path = os.path.join(OUTPUT_DIR, f"{prefix}_per_treatment_probs.csv")
        simple_df.to_csv(simple_path, index=False, encoding='utf-8-sig')
        print(f"[INFO] {prefix}简化预测结果已保存到: {simple_path}")
        
        # 生成热力图矩阵
        heatmap_path = self.heatmap_generator.generate_heatmap_matrix(detailed_df, prefix)
        
        # 生成推荐报告
        rec_path = self.recommendation_generator.generate_recommendations(result_data, prefix)
        
        return {
            'detailed': detailed_path,
            'simple': simple_path,
            'heatmap': heatmap_path,
            'recommendations': rec_path
        }
    
    def _create_simplified_table(self, result_data: Dict[str, Any]) -> pd.DataFrame:
        """创建简化的预测结果表"""
        # 选择关键列
        key_columns = [
            "患者ID", "影像ID", "实际pCR结果", "实际治疗方案", 
            "实际方案预测pCR率", "推荐治疗方案", "推荐方案预测pCR率", "提升空间"
        ]
        
        # 添加所有治疗方案的预测pCR率
        for treatment_name in self.keep:
            key_columns.append(f"pCR率[{treatment_name}]")
        
        return pd.DataFrame({col: result_data[col] for col in key_columns})
    
    def process_full_data_predictions(self, X_all: np.ndarray, y_all: np.ndarray, 
                                    all_probs: np.ndarray, t_all: np.ndarray) -> Dict[str, str]:
        """
        处理全数据预测结果
        
        Args:
            X_all: 全数据特征矩阵
            y_all: 全数据真实标签
            all_probs: 全数据预测概率矩阵
            t_all: 全数据治疗方案索引
            
        Returns:
            包含生成文件路径的字典
        """
        # 将治疗方案名称转换为索引
        t_all_indices = np.array([self.t_map[treatment] for treatment in t_all])
        
        # 生成全数据预测结果
        return self.process_predictions(X_all, y_all, all_probs, t_all_indices, "all_data")

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MNISTPreprocessor:
    """MNIST 資料前處理與 EDA"""
    
    def __init__(self, train_path, test_path=None):
        """
        載入資料
        
        Parameters:
        -----------
        train_path : str
            訓練資料路徑（有 label）
        test_path : str, optional
            測試資料路徑（無 label，用於 Kaggle 提交）
        """
        print("📂 載入訓練資料中...")
        self.train_df = pd.read_csv(train_path)
        self.train_labels = self.train_df.iloc[:, 0].values
        self.train_images = self.train_df.iloc[:, 1:].values
        print(f"✓ 訓練資料載入完成：{len(self.train_df)} 筆")
        
        # 載入測試資料（如果有提供）
        if test_path:
            print(f"📂 載入測試資料中...")
            self.test_df = pd.read_csv(test_path)
            self.test_images = self.test_df.values  # 測試資料沒有 label
            print(f"✓ 測試資料載入完成：{len(self.test_df)} 筆")
        else:
            self.test_df = None
            self.test_images = None
    
    # ==================== EDA 功能 ====================
    
    def check_basic_info(self):
        """基本資訊檢查"""
        print("\n" + "="*60)
        print("📊 資料基本資訊")
        print("="*60)
        print(f"訓練資料筆數：{len(self.train_df)}")
        print(f"特徵數量：{self.train_images.shape[1]} (應為 784)")
        print(f"標籤範圍：{self.train_labels.min()} ~ {self.train_labels.max()}")
        print(f"像素值範圍：{self.train_images.min()} ~ {self.train_images.max()}")
        
        if self.test_images is not None:
            print(f"\n測試資料筆數：{len(self.test_df)}")
            print(f"測試資料像素值範圍：{self.test_images.min()} ~ {self.test_images.max()}")
        
        # 檢查缺失值
        train_missing = self.train_df.isnull().sum().sum()
        print(f"\n訓練資料缺失值：{train_missing}")
        
        if self.test_images is not None:
            test_missing = self.test_df.isnull().sum().sum()
            print(f"測試資料缺失值：{test_missing}")
        
        # 記憶體使用
        train_memory = self.train_df.memory_usage(deep=True).sum() / 1024**2
        print(f"\n訓練資料記憶體使用：{train_memory:.2f} MB")
        
        if self.test_images is not None:
            test_memory = self.test_df.memory_usage(deep=True).sum() / 1024**2
            print(f"測試資料記憶體使用：{test_memory:.2f} MB")
    
    def check_label_distribution(self):
        """檢查標籤分佈（類別平衡）"""
        print("\n" + "="*60)
        print("🔢 標籤分佈分析")
        print("="*60)
        
        # 統計各數字出現次數
        label_counts = pd.Series(self.train_labels).value_counts().sort_index()
        print(label_counts)
        
        # 計算不平衡程度
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\n不平衡比例：{imbalance_ratio:.2f} (理想值接近 1.0)")
        
        if imbalance_ratio > 1.5:
            print("⚠️  警告：資料不平衡，考慮使用 class_weight 或重採樣")
        else:
            print("✓ 資料分佈均衡")
        
        # 視覺化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 長條圖
        label_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('各數字出現次數', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('數字')
        axes[0].set_ylabel('數量')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 圓餅圖
        axes[1].pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
                    startangle=90, colors=plt.cm.tab10.colors)
        axes[1].set_title('數字比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return label_counts
    
    def check_pixel_statistics(self):
        """像素值統計分析"""
        print("\n" + "="*60)
        print("🎨 像素值統計分析")
        print("="*60)
        
        # 整體統計
        print("【訓練資料】")
        print(f"平均像素值：{self.train_images.mean():.2f}")
        print(f"像素值標準差：{self.train_images.std():.2f}")
        print(f"非零像素比例：{(self.train_images > 0).sum() / self.train_images.size * 100:.2f}%")
        
        if self.test_images is not None:
            print("\n【測試資料】")
            print(f"平均像素值：{self.test_images.mean():.2f}")
            print(f"像素值標準差：{self.test_images.std():.2f}")
            print(f"非零像素比例：{(self.test_images > 0).sum() / self.test_images.size * 100:.2f}%")
        
        # 視覺化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 像素值分佈直方圖
        axes[0].hist(self.train_images.flatten(), bins=50, color='steelblue', alpha=0.7, label='Train')
        if self.test_images is not None:
            axes[0].hist(self.test_images.flatten(), bins=50, color='orange', alpha=0.5, label='Test')
        axes[0].set_title('像素值分佈', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('像素值')
        axes[0].set_ylabel('頻率')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 平均像素圖（看哪些位置常有筆畫）
        mean_image = self.train_images.mean(axis=0).reshape(28, 28)
        im = axes[1].imshow(mean_image, cmap='hot')
        axes[1].set_title('平均像素熱力圖（筆畫集中區域）', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_samples_by_label(self, samples_per_label=5):
        """每個數字顯示多個範例"""
        print(f"\n📸 顯示每個數字的 {samples_per_label} 個範例...")
        
        fig, axes = plt.subplots(10, samples_per_label, figsize=(15, 20))
        fig.suptitle('每個數字的範例', fontsize=16, fontweight='bold')
        
        for digit in range(10):
            # 找出該數字的所有索引
            digit_indices = np.where(self.train_labels == digit)[0]
            
            # 隨機選擇範例
            selected_indices = np.random.choice(digit_indices, 
                                               size=min(samples_per_label, len(digit_indices)),
                                               replace=False)
            
            for i, idx in enumerate(selected_indices):
                image = self.train_images[idx].reshape(28, 28)
                axes[digit, i].imshow(image, cmap='gray')
                axes[digit, i].axis('off')
                if i == 0:
                    axes[digit, i].set_ylabel(f'數字 {digit}', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def detect_problematic_images(self, threshold=10):
        """偵測可能有問題的圖片（太暗或太亮）"""
        print(f"\n🔍 偵測問題圖片（平均像素值 < {threshold}）...")
        
        # 計算每張圖的平均像素值
        mean_pixels = self.train_images.mean(axis=1)
        
        # 找出太暗的圖（可能是空白）
        dark_images = np.where(mean_pixels < threshold)[0]
        
        print(f"✓ 找到 {len(dark_images)} 張疑似問題圖片")
        
        if len(dark_images) > 0:
            # 顯示前 10 張
            num_show = min(10, len(dark_images))
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle('疑似問題圖片', fontsize=16, fontweight='bold')
            
            for i in range(num_show):
                idx = dark_images[i]
                row, col = i // 5, i % 5
                image = self.train_images[idx].reshape(28, 28)
                axes[row, col].imshow(image, cmap='gray')
                axes[row, col].set_title(f'Label: {self.train_labels[idx]}\nMean: {mean_pixels[idx]:.1f}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return dark_images
    
    # ==================== 正規化功能 ====================
    
    def normalize_data(self, images, method='minmax'):
        """
        正規化資料
        
        Parameters:
        -----------
        images : numpy array
            要正規化的影像資料
        method : str
            'minmax' : 縮放到 [0, 1]
            'standardize' : 標準化 (mean=0, std=1)
        """
        if method == 'minmax':
            # Min-Max 正規化：除以 255
            normalized = images.astype(np.float32) / 255.0
        elif method == 'standardize':
            # 標準化：(x - mean) / std
            mean = images.mean()
            std = images.std()
            normalized = (images.astype(np.float32) - mean) / std
        else:
            raise ValueError("method 必須是 'minmax' 或 'standardize'")
        
        return normalized
    
    def prepare_for_cnn(self, validation_size=0.2, normalize_method='minmax', random_state=42):
        """
        準備 CNN 訓練資料
        
        Parameters:
        -----------
        validation_size : float
            驗證集比例（從訓練集切出）
        normalize_method : str
            正規化方法
        random_state : int
            隨機種子
        
        Returns:
        --------
        X_train, X_val, y_train, y_val, X_test
        """
        print("\n" + "="*60)
        print("🚀 準備 CNN 訓練資料")
        print("="*60)
        
        # 1. 正規化訓練資料
        print(f"\n🔧 正規化訓練資料（方法：{normalize_method}）...")
        normalized_train = self.normalize_data(self.train_images, method=normalize_method)
        print(f"✓ 訓練資料正規化完成，範圍：[{normalized_train.min():.3f}, {normalized_train.max():.3f}]")
        
        # 2. Reshape 成 CNN 輸入格式 (n_samples, 28, 28, 1)
        reshaped_train = normalized_train.reshape(-1, 28, 28, 1)
        print(f"✓ 訓練資料 Reshape 完成，形狀：{reshaped_train.shape}")
        
        # 3. 切分訓練集與驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            reshaped_train, self.train_labels, 
            test_size=validation_size, 
            random_state=random_state,
            stratify=self.train_labels  # 確保訓練集和驗證集的標籤分佈一致
        )
        
        print(f"\n✓ 資料切分完成")
        print(f"  - 訓練集：{len(X_train)} 筆 ({(1-validation_size)*100:.0f}%)")
        print(f"  - 驗證集：{len(X_val)} 筆 ({validation_size*100:.0f}%)")
        
        # 4. 處理測試資料（如果有）
        X_test = None
        if self.test_images is not None:
            print(f"\n🔧 正規化測試資料...")
            normalized_test = self.normalize_data(self.test_images, method=normalize_method)
            X_test = normalized_test.reshape(-1, 28, 28, 1)
            print(f"✓ 測試資料處理完成")
            print(f"  - 測試集：{len(X_test)} 筆")
        
        # 顯示最終形狀
        print(f"\n📊 最終資料形狀確認：")
        print(f"  X_train.shape: {X_train.shape}")
        print(f"  X_val.shape: {X_val.shape}")
        print(f"  y_train.shape: {y_train.shape}")
        print(f"  y_val.shape: {y_val.shape}")
        if X_test is not None:
            print(f"  X_test.shape: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test


# ==================== 主程式 ====================
if __name__ == "__main__":
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'  # Kaggle 提供的測試資料
    
    print("="*60)
    print("🎯 MNIST 資料前處理與 EDA 完整流程")
    print("="*60)
    
    # 初始化（載入訓練+測試資料）
    preprocessor = MNISTPreprocessor(TRAIN_PATH, TEST_PATH)
    
    # ===== EDA 階段 =====
    print("\n\n" + "🔍 開始 EDA 分析".center(60, "="))
    
    # 1. 基本資訊
    preprocessor.check_basic_info()
    
    # 2. 標籤分佈
    label_dist = preprocessor.check_label_distribution()
    
    # 3. 像素統計
    preprocessor.check_pixel_statistics()
    
    # 4. 顯示範例
    preprocessor.visualize_samples_by_label(samples_per_label=5)
    
    # 5. 偵測問題圖片
    problematic = preprocessor.detect_problematic_images(threshold=10)
    
    # ===== 正規化與準備訓練資料 =====
    print("\n\n" + "🚀 準備訓練資料".center(60, "="))
    
    X_train, X_val, y_train, y_val, X_test = preprocessor.prepare_for_cnn(
        validation_size=0.2,  # 從訓練集切 20% 當驗證集
        normalize_method='minmax',
        random_state=42
    )
    
    # 儲存處理後的資料
    print("\n💾 儲存處理後的資料...")
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    if X_test is not None:
        np.save('X_test.npy', X_test)
    print("✓ 資料已儲存")
    
    print("\n" + "="*60)
    print("✅ 前處理完成！資料切分策略：")
    print("="*60)
    print(f"📌 Train (80%)：用於訓練模型")
    print(f"📌 Validation (20%)：用於調整超參數、監控過擬合")
    print(f"📌 Test (Kaggle)：用於最終提交預測結果")
    print("="*60)
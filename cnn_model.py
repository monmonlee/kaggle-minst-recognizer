import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import pandas as pd
from datetime import datetime

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定隨機種子（確保結果可重現）
np.random.seed(42)
tf.random.set_seed(42)

class MNISTCNNTrainer:
    """MNIST CNN 訓練器"""
    
    def __init__(self, model_name="MNIST_CNN"):
        """初始化訓練器"""
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def load_preprocessed_data(self):
        """載入已前處理的資料"""
        print("📂 載入前處理資料...")
        
        self.X_train = np.load('X_train.npy')
        self.X_val = np.load('X_val.npy')
        self.y_train = np.load('y_train.npy')
        self.y_val = np.load('y_val.npy')
        self.X_test = np.load('X_test.npy')
        
        print(f"✓ 資料載入完成")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_val: {self.X_val.shape}")
        print(f"  X_test: {self.X_test.shape}")
        
        # 轉換標籤為 One-Hot Encoding（CNN 需要）
        self.y_train_categorical = to_categorical(self.y_train, 10)
        self.y_val_categorical = to_categorical(self.y_val, 10)
        
        print(f"✓ 標籤轉換完成（One-Hot Encoding）")
        print(f"  y_train: {self.y_train_categorical.shape}")
        
    def build_cnn_model(self, architecture='standard'):
        """
        建立 CNN 模型
        
        Parameters:
        -----------
        architecture : str
            'standard' - 標準 CNN
            'deep' - 更深的 CNN（更多層）
            'lightweight' - 輕量級 CNN（參數少）
        """
        print(f"\n🏗️  建立 CNN 模型（架構：{architecture}）...")
        
        if architecture == 'standard':
            model = self._build_standard_cnn()
        elif architecture == 'deep':
            model = self._build_deep_cnn()
        elif architecture == 'lightweight':
            model = self._build_lightweight_cnn()
        else:
            raise ValueError("architecture 必須是 'standard', 'deep', 或 'lightweight'")
        
        self.model = model
        
        # 顯示模型架構
        print("\n📊 模型架構摘要：")
        model.summary()
        
        # 計算參數量
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        print(f"\n✓ 可訓練參數數量：{trainable_params:,}")
        
        return model
    
    def _build_standard_cnn(self):
        """標準 CNN 架構（根據你的計畫）"""
        model = models.Sequential([
            # 第一層卷積 + 池化
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # 第二層卷積 + 池化
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # 第三層卷積（增加深度）
            layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
            
            # 展平層
            layers.Flatten(name='flatten'),
            
            # 全連接層
            layers.Dense(64, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout'),  # 防止過擬合
            
            # 輸出層（10 個類別）
            layers.Dense(10, activation='softmax', name='output')
        ], name='Standard_CNN')
        
        return model
    
    def _build_deep_cnn(self):
        """更深的 CNN 架構（適合複雜特徵學習）"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ], name='Deep_CNN')
        
        return model
    
    def _build_lightweight_cnn(self):
        """輕量級 CNN（適合快速訓練）"""
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ], name='Lightweight_CNN')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        編譯模型
        
        Parameters:
        -----------
        learning_rate : float
            學習率
        """
        print(f"\n⚙️  編譯模型（學習率：{learning_rate}）...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ 模型編譯完成")
    
    def train(self, epochs=20, batch_size=128, use_callbacks=True):
        """
        訓練模型
        
        Parameters:
        -----------
        epochs : int
            訓練週期數
        batch_size : int
            批次大小
        use_callbacks : bool
            是否使用回調函數（Early Stopping, Model Checkpoint）
        """
        print("\n" + "="*60)
        print("🚀 開始訓練模型")
        print("="*60)
        print(f"訓練參數：")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - 訓練樣本數: {len(self.X_train)}")
        print(f"  - 驗證樣本數: {len(self.X_val)}")
        print("="*60)
        
        # 設定回調函數
        callback_list = []
        
        if use_callbacks:
            # Early Stopping：驗證損失不再下降時提前停止
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stopping)
            
            # Model Checkpoint：儲存最佳模型
            checkpoint = callbacks.ModelCheckpoint(
                f'{self.model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
            
            # Learning Rate Scheduler：動態調整學習率
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # 開始訓練
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.X_train, self.y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=callback_list,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print(f"✅ 訓練完成！總耗時：{training_time:.2f} 秒")
        print("="*60)
        
        return self.history
    
    def plot_training_history(self):
        """視覺化訓練過程"""
        print("\n📊 繪製訓練歷史...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 準確率曲線
        axes[0].plot(self.history.history['accuracy'], label='訓練準確率', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='驗證準確率', linewidth=2)
        axes[0].set_title('模型準確率變化', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('準確率')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 損失函數曲線
        axes[1].plot(self.history.history['loss'], label='訓練損失', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='驗證損失', linewidth=2)
        axes[1].set_title('模型損失變化', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('損失值')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 圖表已儲存：{self.model_name}_training_history.png")
    
    def evaluate(self):
        """評估模型在驗證集上的表現"""
        print("\n" + "="*60)
        print("📈 評估模型表現")
        print("="*60)
        
        # 在驗證集上評估
        val_loss, val_accuracy = self.model.evaluate(
            self.X_val, self.y_val_categorical, 
            verbose=0
        )
        
        print(f"\n驗證集結果：")
        print(f"  - 損失值：{val_loss:.4f}")
        print(f"  - 準確率：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # 預測
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 分類報告
        print(f"\n📊 詳細分類報告：")
        print(classification_report(self.y_val, y_pred, 
                                   target_names=[str(i) for i in range(10)]))
        
        return val_accuracy, y_pred
    
    def plot_confusion_matrix(self, y_pred):
        """繪製混淆矩陣"""
        print("\n🎨 繪製混淆矩陣...")
        
        # 計算混淆矩陣
        cm = confusion_matrix(self.y_val, y_pred)
        
        # 繪製
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始計數
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=range(10), yticklabels=range(10))
        axes[0].set_title('混淆矩陣（絕對數量）', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('預測標籤')
        axes[0].set_ylabel('真實標籤')
        
        # 正規化（百分比）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=range(10), yticklabels=range(10))
        axes[1].set_title('混淆矩陣（比例）', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('預測標籤')
        axes[1].set_ylabel('真實標籤')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 分析最容易混淆的數字對
        print("\n🔍 最容易混淆的數字對（Top 5）：")
        errors = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i][j] > 0:
                    errors.append((i, j, cm[i][j], cm_normalized[i][j]))
        
        errors.sort(key=lambda x: x[2], reverse=True)
        for rank, (true, pred, count, ratio) in enumerate(errors[:5], 1):
            print(f"  {rank}. 真實={true}, 預測={pred}: {count} 次 ({ratio:.2%})")
        
        print(f"\n✓ 圖表已儲存：{self.model_name}_confusion_matrix.png")
    
    def visualize_predictions(self, num_samples=20):
        """視覺化預測結果（包含錯誤案例）"""
        print(f"\n🖼️  視覺化預測結果（顯示 {num_samples} 個樣本）...")
        
        # 隨機選擇樣本
        indices = np.random.choice(len(self.X_val), num_samples, replace=False)
        
        # 預測
        predictions = self.model.predict(self.X_val[indices], verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = self.y_val[indices]
        
        # 繪圖
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        fig.suptitle('預測結果視覺化（綠色=正確，紅色=錯誤）', 
                     fontsize=16, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < num_samples:
                image = self.X_val[indices[idx]].reshape(28, 28)
                true_label = true_labels[idx]
                pred_label = predicted_labels[idx]
                confidence = predictions[idx][pred_label] * 100
                
                ax.imshow(image, cmap='gray')
                
                # 正確=綠色，錯誤=紅色
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'真實:{true_label} 預測:{pred_label}\n信心度:{confidence:.1f}%',
                           color=color, fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 圖表已儲存：{self.model_name}_predictions.png")
    
    def predict_test_set(self, output_path='submission.csv'):
        """預測測試集並產生 Kaggle 提交檔案"""
        print("\n" + "="*60)
        print("🎯 預測測試集")
        print("="*60)
        
        # 預測
        print("正在預測...")
        test_predictions = self.model.predict(self.X_test, verbose=1)
        test_labels = np.argmax(test_predictions, axis=1)
        
        # 產生提交檔案
        submission = pd.DataFrame({
            'ImageId': range(1, len(test_labels) + 1),
            'Label': test_labels
        })
        
        submission.to_csv(output_path, index=False)
        
        print(f"\n✓ 預測完成！")
        print(f"✓ 提交檔案已儲存：{output_path}")
        print(f"✓ 總共預測：{len(test_labels)} 筆資料")
        print(f"\n預測標籤分佈：")
        print(pd.Series(test_labels).value_counts().sort_index())
        
        return submission
    
    def save_model(self, filepath=None):
        """儲存模型"""
        if filepath is None:
            filepath = f'{self.model_name}_final.keras'
        
        self.model.save(filepath)
        print(f"\n💾 模型已儲存：{filepath}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    print("="*60)
    print("🎯 MNIST CNN 訓練完整流程")
    print("="*60)
    
    # 1. 初始化訓練器
    trainer = MNISTCNNTrainer(model_name="MNIST_CNN_Standard")
    
    # 2. 載入資料
    trainer.load_preprocessed_data()
    
    # 3. 建立模型（可選擇：'standard', 'deep', 'lightweight'）
    trainer.build_cnn_model(architecture='standard')
    
    # 4. 編譯模型
    trainer.compile_model(learning_rate=0.001)
    
    # 5. 訓練模型
    history = trainer.train(
        epochs=20,
        batch_size=128,
        use_callbacks=True
    )
    
    # 6. 視覺化訓練過程
    trainer.plot_training_history()
    
    # 7. 評估模型
    val_accuracy, y_pred = trainer.evaluate()
    
    # 8. 繪製混淆矩陣
    trainer.plot_confusion_matrix(y_pred)
    
    # 9. 視覺化預測結果
    trainer.visualize_predictions(num_samples=20)
    
    # 10. 預測測試集並產生提交檔案
    submission = trainer.predict_test_set(output_path='submission.csv')
    
    # 11. 儲存模型
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✅ 所有流程完成！")
    print("="*60)
    print("\n產生的檔案：")
    print("  📊 MNIST_CNN_Standard_training_history.png")
    print("  📊 MNIST_CNN_Standard_confusion_matrix.png")
    print("  📊 MNIST_CNN_Standard_predictions.png")
    print("  💾 MNIST_CNN_Standard_best.keras")
    print("  💾 MNIST_CNN_Standard_final.keras")
    print("  📄 submission.csv")
    print("="*60)
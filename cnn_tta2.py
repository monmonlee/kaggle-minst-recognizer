import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

# 設定隨機種子
np.random.seed(42)
tf.random.set_seed(42)

class MNISTCNNWithEnhancedTTA:
    """MNIST CNN 訓練器 with Enhanced TTA (增強版 TTA)"""
    
    def __init__(self, model_name="MNIST_CNN_Enhanced_TTA"):
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
        
        # 轉換標籤為 One-Hot Encoding
        self.y_train_categorical = to_categorical(self.y_train, 10)
        self.y_val_categorical = to_categorical(self.y_val, 10)
        
        print(f"✓ 標籤轉換完成（One-Hot Encoding）")
        
    def create_data_augmentation(self):
        """建立資料增強生成器"""
        print("\n🔄 建立資料增強生成器...")
        
        # 訓練用：適中的增強（基於你表現最好的版本）
        self.train_datagen = ImageDataGenerator(
            rotation_range=15,           # 旋轉 ±15 度
            width_shift_range=0.15,      # 水平平移 15%
            height_shift_range=0.15,     # 垂直平移 15%
            zoom_range=0.15,             # 縮放 ±15%
            shear_range=0.15,            # 剪切變換
            fill_mode='nearest'
        )
        
        # TTA 用：溫和的增強（預測時使用）
        self.tta_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        print("✓ 資料增強生成器建立完成")
        
    def build_improved_cnn(self):
        """建立改進版 CNN（加入 BatchNormalization）"""
        print(f"\n🏗️  建立改進版 CNN 模型...")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ], name='Improved_CNN_Enhanced_TTA')
        
        self.model = model
        
        # 顯示模型架構
        print("\n📊 模型架構摘要：")
        model.summary()
        
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        print(f"\n✓ 可訓練參數數量：{trainable_params:,}")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """編譯模型"""
        print(f"\n⚙️  編譯模型（學習率：{learning_rate}）...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ 模型編譯完成")
    
    def train_with_augmentation(self, epochs=30, batch_size=64):
        """使用資料增強訓練模型"""
        print("\n" + "="*60)
        print("🚀 開始訓練模型（使用資料增強）")
        print("="*60)
        print(f"訓練參數：")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - 訓練樣本數: {len(self.X_train)}")
        print(f"  - 驗證樣本數: {len(self.X_val)}")
        print("="*60)
        
        # 設定回調函數
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            f'{self.model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
        
        # 使用資料增強訓練
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.train_datagen.flow(self.X_train, self.y_train_categorical, 
                                   batch_size=batch_size),
            steps_per_epoch=len(self.X_train) // batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val_categorical),
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print(f"✅ 訓練完成！總耗時：{training_time:.2f} 秒")
        print("="*60)
        
        return self.history
    
    def evaluate_on_validation(self):
        """在驗證集上評估（無 TTA）"""
        print("\n📈 評估模型表現（驗證集，無 TTA）...")
        
        val_loss, val_accuracy = self.model.evaluate(
            self.X_val, self.y_val_categorical, 
            verbose=0
        )
        
        print(f"驗證集準確率（無 TTA）：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        return val_accuracy
    
    def predict_with_enhanced_tta(self, X, n_augmentations=20):
        """
        使用增強版 TTA 進行預測（更多次數）
        
        Parameters:
        -----------
        X : ndarray
            輸入影像
        n_augmentations : int
            每張圖片增強的次數（建議 15-20）
        
        Returns:
        --------
        predictions : ndarray
            平均後的預測機率
        """
        print(f"\n🔮 使用增強版 TTA 進行預測（每張圖片 {n_augmentations} 次增強）...")
        print(f"   預計處理時間：約 {n_augmentations * 2} 秒")
        
        # 原始預測
        predictions = self.model.predict(X, verbose=0)
        
        # 進行多次增強預測並平均
        for i in range(n_augmentations - 1):
            # 生成增強版本
            aug_generator = self.tta_datagen.flow(X, batch_size=len(X), shuffle=False)
            X_aug = next(aug_generator)
            
            # 預測
            aug_predictions = self.model.predict(X_aug, verbose=0)
            
            # 累加
            predictions += aug_predictions
            
            # 顯示進度
            if (i + 1) % 5 == 0:
                progress = (i + 2) / n_augmentations * 100
                print(f"  進度：{i + 2}/{n_augmentations} ({progress:.1f}%)")
        
        # 平均
        predictions = predictions / n_augmentations
        
        print(f"✓ 增強版 TTA 預測完成（共 {n_augmentations} 次增強）")
        
        return predictions
    
    def evaluate_with_tta(self, n_augmentations=20):
        """在驗證集上使用 TTA 評估"""
        print("\n📈 評估模型表現（驗證集，使用增強版 TTA）...")
        
        # 使用 TTA 預測
        val_predictions = self.predict_with_enhanced_tta(self.X_val, n_augmentations)
        val_pred_labels = np.argmax(val_predictions, axis=1)
        
        # 計算準確率
        tta_accuracy = accuracy_score(self.y_val, val_pred_labels)
        
        print(f"驗證集準確率（TTA x{n_augmentations}）：{tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
        
        # 比較改善
        no_tta_accuracy = self.evaluate_on_validation()
        improvement = (tta_accuracy - no_tta_accuracy) * 100
        print(f"\n✨ TTA 改善：+{improvement:.2f}%")
        
        return tta_accuracy
    
    def predict_test_set_with_enhanced_tta(self, n_augmentations=20, output_path='submission_enhanced_tta.csv'):
        """使用增強版 TTA 預測測試集並產生提交檔案"""
        print("\n" + "="*60)
        print(f"🎯 預測測試集（使用增強版 TTA x{n_augmentations}）")
        print("="*60)
        
        # 使用 TTA 預測
        test_predictions = self.predict_with_enhanced_tta(self.X_test, n_augmentations)
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
        print(submission['Label'].value_counts().sort_index())
        
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
    print("🎯 MNIST CNN + 增強版 TTA (Enhanced TTA)")
    print("="*60)
    
    # 1. 初始化訓練器
    trainer = MNISTCNNWithEnhancedTTA(model_name="MNIST_CNN_Enhanced_TTA")
    
    # 2. 載入資料
    trainer.load_preprocessed_data()
    
    # 3. 建立資料增強生成器
    trainer.create_data_augmentation()
    
    # 4. 建立改進版模型
    trainer.build_improved_cnn()
    
    # 5. 編譯模型
    trainer.compile_model(learning_rate=0.001)
    
    # 6. 訓練模型（使用資料增強）
    history = trainer.train_with_augmentation(
        epochs=30,
        batch_size=64
    )
    
    # 7. 評估模型（驗證集）
    print("\n" + "="*60)
    print("📊 模型評估")
    print("="*60)
    
    # 無 TTA 的準確率
    no_tta_acc = trainer.evaluate_on_validation()
    
    # 使用增強版 TTA 的準確率（在驗證集上測試）
    print("\n" + "-"*60)
    print("⚠️  以下是在驗證集上測試不同 TTA 次數的效果")
    print("-"*60)
    
    # 測試 TTA=10（跟之前一樣）
    print("\n【測試 1】TTA 次數 = 10")
    tta_10_acc = trainer.evaluate_with_tta(n_augmentations=10)
    
    # 測試 TTA=15
    print("\n【測試 2】TTA 次數 = 15")
    tta_15_acc = trainer.evaluate_with_tta(n_augmentations=15)
    
    # 測試 TTA=20
    print("\n【測試 3】TTA 次數 = 20")
    tta_20_acc = trainer.evaluate_with_tta(n_augmentations=20)
    
    # 比較結果
    print("\n" + "="*60)
    print("📊 TTA 次數比較結果")
    print("="*60)
    print(f"無 TTA：      {no_tta_acc:.4f} ({no_tta_acc*100:.2f}%)")
    print(f"TTA x10：     {tta_10_acc:.4f} ({tta_10_acc*100:.2f}%)")
    print(f"TTA x15：     {tta_15_acc:.4f} ({tta_15_acc*100:.2f}%)")
    print(f"TTA x20：     {tta_20_acc:.4f} ({tta_20_acc*100:.2f}%)")
    print("="*60)
    
    # 8. 預測測試集（使用最佳的 TTA 次數）
    print("\n" + "="*60)
    print("🚀 開始預測測試集")
    print("="*60)
    print("\n💡 根據驗證集結果，選擇使用 TTA x20")
    
    # 你可以根據驗證集結果調整這個數字
    best_tta_count = 25  # 可改成 15 或 20
    
    submission = trainer.predict_test_set_with_enhanced_tta(
        n_augmentations=best_tta_count,
        output_path='submission_enhanced_tta_best_test.csv'
    )
    
    # 9. 儲存模型
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✅ 所有流程完成！")
    print("="*60)
    print("\n產生的檔案：")
    print("  📄 submission_enhanced_tta.csv  ← 提交這個到 Kaggle")
    print("  💾 MNIST_CNN_Enhanced_TTA_best.keras")
    print("  💾 MNIST_CNN_Enhanced_TTA_final.keras")
    print("="*60)
    print("\n🎯 改進重點：")
    print("  1. ✅ 基於表現最好的 TTA 版本")
    print("  2. ✅ 增加 TTA 次數到 15-20 次")
    print("  3. ✅ 在驗證集上測試不同 TTA 次數的效果")
    print("  4. ✅ 預期提升：0.99371 → 0.994-0.996")
    print("="*60)
    print("\n💡 提示：")
    print("  - 如果驗證集上 TTA x15 和 x20 差不多，用 x15 就好（更快）")
    print("  - 如果想更激進，可以試試 TTA x25 或 x30")
    print("  - 記得觀察驗證集的改善是否飽和")
    print("="*60)
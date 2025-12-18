import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter, map_coordinates

# 設定隨機種子
np.random.seed(42)
tf.random.set_seed(42)

class MNISTCNNWithElasticTTA:
    """MNIST CNN 訓練器 with Elastic Transform + TTA"""
    
    def __init__(self, model_name="MNIST_CNN_Elastic_TTA"):
        """初始化訓練器"""
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def elastic_transform(self, image, alpha=36, sigma=4, random_state=None):
        """
        彈性變形 (Elastic Transform)
        
        Parameters:
        -----------
        image : ndarray
            輸入影像 (28, 28, 1)
        alpha : float
            變形強度（數值越大變形越明顯）
        sigma : float
            平滑程度（高斯濾波的標準差）
        random_state : int
            隨機種子
            
        Returns:
        --------
        變形後的影像
        
        原理：
        1. 生成隨機位移場（每個像素往哪個方向移動）
        2. 用高斯濾波平滑位移場（讓變形看起來自然）
        3. 根據位移場重新映射像素位置
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        
        shape = image.shape[:2]  # (28, 28)
        
        # 生成隨機位移場
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        
        # 生成座標網格
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # 應用變形（對每個通道）
        if len(image.shape) == 3:
            distorted_image = np.zeros_like(image)
            for i in range(image.shape[2]):
                distorted_image[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            distorted_image = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
            
        return distorted_image
    
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
        
    def create_elastic_augmentation_generator(self, X, y, batch_size=64):
        """
        建立包含 Elastic Transform 的資料生成器
        
        這個生成器會：
        1. 先做基礎增強（旋轉、平移等）
        2. 然後有 50% 機率做 Elastic Transform
        """
        # 基礎增強
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        # 生成基礎增強的資料
        generator = datagen.flow(X, y, batch_size=batch_size)
        
        while True:
            # 取得一個 batch
            X_batch, y_batch = next(generator)
            
            # 對 batch 中的每張圖片，有 50% 機率做 Elastic Transform
            X_elastic = np.zeros_like(X_batch)
            for i in range(len(X_batch)):
                if np.random.random() > 0.5:
                    # 做 Elastic Transform
                    X_elastic[i] = self.elastic_transform(
                        X_batch[i],
                        alpha=36,
                        sigma=4
                    )
                else:
                    # 不做變形
                    X_elastic[i] = X_batch[i]
            
            yield X_elastic, y_batch
    
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
        ], name='Improved_CNN_Elastic')
        
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
    
    def train_with_elastic_augmentation(self, epochs=35, batch_size=64):
        """使用 Elastic Transform 訓練模型"""
        print("\n" + "="*60)
        print("🚀 開始訓練模型（使用 Elastic Transform）")
        print("="*60)
        print(f"訓練參數：")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - 訓練樣本數: {len(self.X_train)}")
        print(f"  - 驗證樣本數: {len(self.X_val)}")
        print(f"  - 使用增強：旋轉、平移、縮放、剪切 + Elastic Transform")
        print("="*60)
        
        # 設定回調函數
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
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
        
        # 建立包含 Elastic Transform 的生成器
        train_generator = self.create_elastic_augmentation_generator(
            self.X_train, 
            self.y_train_categorical,
            batch_size=batch_size
        )
        
        # 訓練
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_generator,
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
    
    def predict_with_tta(self, X, n_augmentations=10):
        """
        使用 Test Time Augmentation 進行預測
        
        包含：基礎增強 + Elastic Transform
        """
        print(f"\n🔮 使用 TTA 進行預測（每張圖片 {n_augmentations} 次增強）...")
        
        # 原始預測
        predictions = self.model.predict(X, verbose=0)
        
        # 建立 TTA 生成器
        tta_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # 進行多次增強預測
        for i in range(n_augmentations - 1):
            # 基礎增強
            aug_generator = tta_datagen.flow(X, batch_size=len(X), shuffle=False)
            X_aug = next(aug_generator)
            
            # 有 50% 機率額外做 Elastic Transform
            if np.random.random() > 0.5:
                X_elastic = np.zeros_like(X_aug)
                for j in range(len(X_aug)):
                    X_elastic[j] = self.elastic_transform(X_aug[j], alpha=36, sigma=4)
                X_aug = X_elastic
            
            # 預測
            aug_predictions = self.model.predict(X_aug, verbose=0)
            
            # 累加
            predictions += aug_predictions
            
            if (i + 1) % 3 == 0:
                print(f"  進度：{i + 2}/{n_augmentations}")
        
        # 平均
        predictions = predictions / n_augmentations
        
        print(f"✓ TTA 預測完成")
        
        return predictions
    
    def evaluate_with_tta(self, n_augmentations=10):
        """在驗證集上使用 TTA 評估"""
        print("\n📈 評估模型表現（驗證集，使用 TTA + Elastic）...")
        
        # 使用 TTA 預測
        val_predictions = self.predict_with_tta(self.X_val, n_augmentations)
        val_pred_labels = np.argmax(val_predictions, axis=1)
        
        # 計算準確率
        tta_accuracy = accuracy_score(self.y_val, val_pred_labels)
        
        print(f"驗證集準確率（TTA + Elastic）：{tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
        
        # 比較改善
        no_tta_accuracy = self.evaluate_on_validation()
        improvement = (tta_accuracy - no_tta_accuracy) * 100
        print(f"\n✨ TTA 改善：+{improvement:.2f}%")
        
        return tta_accuracy
    
    def predict_test_set_with_tta(self, n_augmentations=12, output_path='submission_elastic_tta.csv'):
        """使用 TTA + Elastic Transform 預測測試集"""
        print("\n" + "="*60)
        print("🎯 預測測試集（使用 TTA + Elastic Transform）")
        print("="*60)
        
        # 使用 TTA 預測
        test_predictions = self.predict_with_tta(self.X_test, n_augmentations)
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
    print("🎯 MNIST CNN + Elastic Transform + TTA")
    print("="*60)
    
    # 1. 初始化訓練器
    trainer = MNISTCNNWithElasticTTA(model_name="MNIST_CNN_Elastic_TTA")
    
    # 2. 載入資料
    trainer.load_preprocessed_data()
    
    # 3. 建立改進版模型
    trainer.build_improved_cnn()
    
    # 4. 編譯模型
    trainer.compile_model(learning_rate=0.001)
    
    # 5. 訓練模型（使用 Elastic Transform）
    print("\n💡 訓練時會使用：")
    print("   - 旋轉 ±15°")
    print("   - 平移 ±15%")
    print("   - 縮放 ±15%")
    print("   - 剪切變形")
    print("   - Elastic Transform（50% 機率）← 新增！")
    print("")
    
    history = trainer.train_with_elastic_augmentation(
        epochs=35,
        batch_size=64
    )
    
    # 6. 評估模型
    print("\n" + "="*60)
    print("📊 模型評估")
    print("="*60)
    
    # 無 TTA 的準確率
    no_tta_acc = trainer.evaluate_on_validation()
    
    # 使用 TTA + Elastic 的準確率
    tta_acc = trainer.evaluate_with_tta(n_augmentations=10)
    
    # 7. 預測測試集（使用 TTA + Elastic）
    print("\n" + "="*60)
    print("🚀 開始預測測試集")
    print("="*60)
    
    submission = trainer.predict_test_set_with_tta(
        n_augmentations=12,  # 可以調整（10-15 都可以）
        output_path='submission_elastic_tta.csv'
    )
    
    # 8. 儲存模型
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✅ 所有流程完成！")
    print("="*60)
    print("\n產生的檔案：")
    print("  📄 submission_elastic_tta.csv  ← 提交這個到 Kaggle")
    print("  💾 MNIST_CNN_Elastic_TTA_best.keras")
    print("  💾 MNIST_CNN_Elastic_TTA_final.keras")
    print("="*60)
    print("\n🎯 改進重點：")
    print("  1. ✅ 訓練時加入 Elastic Transform（模擬手寫變形）")
    print("  2. ✅ TTA 時也有機率使用 Elastic Transform")
    print("  3. ✅ 預期提升：0.99371 → 0.996-0.998")
    print("="*60)
    print("\n💡 如果還想提升：")
    print("  - 增加 n_augmentations 到 15-20")
    print("  - 訓練更多 epochs（40-50）")
    print("  - 或考慮 Ensemble 多個模型")
    print("="*60)
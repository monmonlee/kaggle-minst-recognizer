import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
import json

# 設定不同的隨機種子來訓練不同的模型
np.random.seed(42)
tf.random.set_seed(42)

class EnsembleModelTrainer:
    """Ensemble 模型訓練器 - 訓練多個不同的模型"""
    
    def __init__(self, model_id=1, random_seed=42):
        """
        初始化訓練器
        
        Parameters:
        -----------
        model_id : int
            模型編號
        random_seed : int
            隨機種子（每個模型用不同的種子）
        """
        self.model_id = model_id
        self.model_name = f"MNIST_Ensemble_Model_{model_id}"
        self.random_seed = random_seed
        self.model = None
        self.history = None
        
        # 設定隨機種子
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
    def load_preprocessed_data(self):
        """載入已前處理的資料"""
        print(f"\n📂 [{self.model_name}] 載入前處理資料...")
        
        self.X_train = np.load('X_train.npy')
        self.X_val = np.load('X_val.npy')
        self.y_train = np.load('y_train.npy')
        self.y_val = np.load('y_val.npy')
        self.X_test = np.load('X_test.npy')
        
        print(f"✓ 資料載入完成")
        
        # 轉換標籤為 One-Hot Encoding
        self.y_train_categorical = to_categorical(self.y_train, 10)
        self.y_val_categorical = to_categorical(self.y_val, 10)
        
    def create_data_augmentation(self, augmentation_type='standard'):
        """
        建立資料增強生成器（不同模型用不同強度）
        
        Parameters:
        -----------
        augmentation_type : str
            'mild' - 溫和增強
            'standard' - 標準增強
            'aggressive' - 激進增強
        """
        print(f"\n🔄 [{self.model_name}] 建立資料增強生成器（{augmentation_type}）...")
        
        if augmentation_type == 'mild':
            # 溫和增強
            self.train_datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                shear_range=0.1,
                fill_mode='nearest'
            )
        elif augmentation_type == 'aggressive':
            # 激進增強
            self.train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:  # standard
            # 標準增強
            self.train_datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.15,
                shear_range=0.15,
                fill_mode='nearest'
            )
        
        print("✓ 資料增強生成器建立完成")
        
    def build_cnn_model(self, architecture='standard'):
        """
        建立 CNN 模型（不同架構）
        
        Parameters:
        -----------
        architecture : str
            'standard' - 標準架構
            'wide' - 更寬的架構（更多 filters）
            'deep' - 更深的架構（更多層）
        """
        print(f"\n🏗️  [{self.model_name}] 建立 CNN 模型（{architecture}）...")
        
        if architecture == 'wide':
            # 更寬的架構（更多 filters）
            model = self._build_wide_cnn()
        elif architecture == 'deep':
            # 更深的架構（更多層）
            model = self._build_deep_cnn()
        else:
            # 標準架構
            model = self._build_standard_cnn()
        
        self.model = model
        
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        print(f"✓ 可訓練參數數量：{trainable_params:,}")
        
        return model
    
    def _build_standard_cnn(self):
        """標準 CNN 架構"""
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
        ], name=f'{self.model_name}_Standard')
        
        return model
    
    def _build_wide_cnn(self):
        """更寬的 CNN 架構（更多 filters）"""
        model = models.Sequential([
            # Block 1 - 更多 filters
            layers.Conv2D(48, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(48, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(96, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(96, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(192, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(384, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(192, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ], name=f'{self.model_name}_Wide')
        
        return model
    
    def _build_deep_cnn(self):
        """更深的 CNN 架構（更多層）"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
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
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
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
        ], name=f'{self.model_name}_Deep')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """編譯模型"""
        print(f"\n⚙️  [{self.model_name}] 編譯模型...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ 模型編譯完成")
    
    def train(self, epochs=30, batch_size=64):
        """訓練模型"""
        print("\n" + "="*60)
        print(f"🚀 [{self.model_name}] 開始訓練")
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
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
        
        # 訓練
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
        print(f"✅ [{self.model_name}] 訓練完成！耗時：{training_time:.2f} 秒")
        print("="*60)
        
        return self.history
    
    def evaluate(self):
        """評估模型"""
        val_loss, val_accuracy = self.model.evaluate(
            self.X_val, self.y_val_categorical, 
            verbose=0
        )
        
        print(f"\n📊 [{self.model_name}] 驗證集準確率：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        return val_accuracy
    
    def save_model(self):
        """儲存模型"""
        filepath = f'{self.model_name}_final.keras'
        self.model.save(filepath)
        print(f"💾 [{self.model_name}] 模型已儲存：{filepath}")
        
        return filepath


# ==================== 主程式：訓練 5 個模型 ====================
if __name__ == "__main__":
    print("="*70)
    print("🎯 Ensemble 訓練腳本 - 訓練多個不同的模型")
    print("="*70)
    
    # 定義 5 個不同的模型配置
    model_configs = [
        {
            'model_id': 1,
            'random_seed': 42,
            'architecture': 'standard',
            'augmentation': 'standard',
            'learning_rate': 0.001,
            'batch_size': 64
        },
        {
            'model_id': 2,
            'random_seed': 123,
            'architecture': 'wide',
            'augmentation': 'standard',
            'learning_rate': 0.001,
            'batch_size': 64
        },
        {
            'model_id': 3,
            'random_seed': 456,
            'architecture': 'deep',
            'augmentation': 'mild',
            'learning_rate': 0.001,
            'batch_size': 64
        },
        {
            'model_id': 4,
            'random_seed': 789,
            'architecture': 'standard',
            'augmentation': 'aggressive',
            'learning_rate': 0.0008,
            'batch_size': 64
        },
        {
            'model_id': 5,
            'random_seed': 999,
            'architecture': 'wide',
            'augmentation': 'standard',
            'learning_rate': 0.0012,
            'batch_size': 48
        }
    ]
    
    # 儲存模型資訊
    models_info = []
    
    # 訓練每個模型
    for i, config in enumerate(model_configs, 1):
        print("\n" + "="*70)
        print(f"🔄 開始訓練模型 {i}/{len(model_configs)}")
        print("="*70)
        print(f"配置：{config}")
        
        # 初始化訓練器
        trainer = EnsembleModelTrainer(
            model_id=config['model_id'],
            random_seed=config['random_seed']
        )
        
        # 載入資料
        trainer.load_preprocessed_data()
        
        # 建立資料增強
        trainer.create_data_augmentation(config['augmentation'])
        
        # 建立模型
        trainer.build_cnn_model(config['architecture'])
        
        # 編譯模型
        trainer.compile_model(config['learning_rate'])
        
        # 訓練模型
        history = trainer.train(epochs=30, batch_size=config['batch_size'])
        
        # 評估模型
        val_accuracy = trainer.evaluate()
        
        # 儲存模型
        model_path = trainer.save_model()
        
        # 記錄模型資訊
        model_info = {
            'model_id': config['model_id'],
            'model_name': trainer.model_name,
            'model_path': model_path,
            'val_accuracy': float(val_accuracy),
            'config': config
        }
        models_info.append(model_info)
        
        print(f"\n✅ 模型 {i} 訓練完成！驗證集準確率：{val_accuracy:.4f}")
    
    # 儲存所有模型的資訊
    with open('ensemble_models_info.json', 'w') as f:
        json.dump(models_info, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ 所有模型訓練完成！")
    print("="*70)
    print("\n📊 模型驗證集準確率總覽：")
    for info in models_info:
        print(f"  {info['model_name']}: {info['val_accuracy']:.4f} ({info['val_accuracy']*100:.2f}%)")
    
    # 計算平均準確率
    avg_accuracy = np.mean([info['val_accuracy'] for info in models_info])
    print(f"\n📈 平均驗證集準確率：{avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    
    print("\n📄 模型資訊已儲存到：ensemble_models_info.json")
    print("\n💡 下一步：執行 ensemble_predict.py 來組合這些模型的預測")
    print("="*70)
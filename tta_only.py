import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

print("="*60)
print("🎯 快速 TTA 預測腳本（不重新訓練）")
print("="*60)

# 1. 載入測試資料
print("\n📂 載入測試資料...")
X_test = np.load('X_test.npy')
print(f"✓ 測試集載入完成：{X_test.shape}")

# 2. 載入已訓練好的模型
print("\n💾 載入已訓練模型...")
# 使用你最好的模型（應該是 Enhanced TTA 的 best 模型）
model = load_model('MNIST_CNN_Enhanced_TTA_best.keras')
print("✓ 模型載入完成")

# 3. 建立 TTA 生成器
print("\n🔄 建立 TTA 資料增強器...")
tta_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
print("✓ TTA 生成器建立完成")

# 4. TTA 預測函數
def predict_with_tta(model, X, n_augmentations=15):
    """
    使用 TTA 進行預測
    
    Parameters:
    -----------
    model : keras Model
        已訓練好的模型
    X : ndarray
        測試資料
    n_augmentations : int
        TTA 次數
    """
    print(f"\n🔮 開始 TTA 預測（{n_augmentations} 次增強）...")
    print(f"   預計時間：約 {n_augmentations * 2} 秒")
    
    # 原始預測
    predictions = model.predict(X, verbose=0)
    print(f"  進度：1/{n_augmentations} (原始預測完成)")
    
    # 進行多次增強預測
    for i in range(n_augmentations - 1):
        # 生成增強版本
        aug_generator = tta_datagen.flow(X, batch_size=len(X), shuffle=False)
        X_aug = next(aug_generator)
        
        # 預測
        aug_predictions = model.predict(X_aug, verbose=0)
        
        # 累加
        predictions += aug_predictions
        
        # 顯示進度
        if (i + 2) % 5 == 0 or (i + 2) == n_augmentations:
            progress = (i + 2) / n_augmentations * 100
            print(f"  進度：{i + 2}/{n_augmentations} ({progress:.1f}%)")
    
    # 平均
    predictions = predictions / n_augmentations
    
    print(f"✓ TTA 預測完成！")
    
    return predictions

# 5. 生成不同 TTA 次數的提交檔案
print("\n" + "="*60)
print("🚀 開始生成不同 TTA 次數的提交檔案")
print("="*60)

tta_counts = [10, 15, 20, 25]  # 你想測試的 TTA 次數

for n_tta in tta_counts:
    print(f"\n【TTA x{n_tta}】")
    
    # TTA 預測
    test_predictions = predict_with_tta(model, X_test, n_augmentations=n_tta)
    test_labels = np.argmax(test_predictions, axis=1)
    
    # 產生提交檔案
    output_filename = f'submission_tta_x{n_tta}.csv'
    submission = pd.DataFrame({
        'ImageId': range(1, len(test_labels) + 1),
        'Label': test_labels
    })
    submission.to_csv(output_filename, index=False)
    
    print(f"✓ 已儲存：{output_filename}")
    print(f"  預測標籤分佈：")
    print(f"  {submission['Label'].value_counts().sort_index().to_dict()}")

print("\n" + "="*60)
print("✅ 所有 TTA 預測完成！")
print("="*60)
print("\n產生的檔案：")
for n_tta in tta_counts:
    print(f"  📄 submission_tta_x{n_tta}.csv")
print("="*60)
print("\n💡 建議：")
print("  1. 先提交 submission_tta_x15.csv（根據你的發現最好）")
print("  2. 如果分數不如預期，再試試其他的")
print("  3. 每個檔案都可以直接提交，不需要重新訓練")
print("="*60)
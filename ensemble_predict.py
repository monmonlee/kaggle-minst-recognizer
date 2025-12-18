import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import json
from collections import Counter

print("="*70)
print("🎯 Ensemble 預測腳本 - 組合多個模型的預測")
print("="*70)

# 1. 載入模型資訊
print("\n📂 載入模型資訊...")
with open('ensemble_models_info.json', 'r') as f:
    models_info = json.load(f)

print(f"✓ 找到 {len(models_info)} 個模型")
for info in models_info:
    print(f"  - {info['model_name']}: 驗證集準確率 {info['val_accuracy']:.4f}")

# 2. 載入測試資料
print("\n📂 載入測試資料...")
X_test = np.load('X_test.npy')
print(f"✓ 測試集載入完成：{X_test.shape}")

# 3. 載入所有模型
print("\n💾 載入所有訓練好的模型...")
models = []
for info in models_info:
    model_path = f"{info['model_name']}_best.keras"
    try:
        model = load_model(model_path)
        models.append({
            'model': model,
            'name': info['model_name'],
            'val_accuracy': info['val_accuracy']
        })
        print(f"  ✓ 載入：{info['model_name']}")
    except Exception as e:
        print(f"  ⚠️  無法載入 {model_path}: {e}")

print(f"\n✓ 成功載入 {len(models)} 個模型")

# 4. 建立 TTA 生成器
print("\n🔄 建立 TTA 資料增強器...")
tta_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
print("✓ TTA 生成器建立完成")

# 5. 定義 TTA 預測函數
def predict_with_tta(model, X, n_augmentations=15):
    """使用 TTA 進行預測"""
    # 原始預測
    predictions = model.predict(X, verbose=0)
    
    # 進行多次增強預測
    for i in range(n_augmentations - 1):
        aug_generator = tta_datagen.flow(X, batch_size=len(X), shuffle=False)
        X_aug = next(aug_generator)
        aug_predictions = model.predict(X_aug, verbose=0)
        predictions += aug_predictions
    
    # 平均
    predictions = predictions / n_augmentations
    
    return predictions

# 6. 使用每個模型進行預測
print("\n" + "="*70)
print("🔮 開始使用每個模型進行 TTA 預測...")
print("="*70)

all_predictions = []
all_pred_labels = []

for i, model_dict in enumerate(models, 1):
    model = model_dict['model']
    model_name = model_dict['name']
    
    print(f"\n[{i}/{len(models)}] {model_name}")
    print(f"  使用 TTA x15 預測...")
    
    # TTA 預測
    predictions = predict_with_tta(model, X_test, n_augmentations=15)
    pred_labels = np.argmax(predictions, axis=1)
    
    all_predictions.append(predictions)
    all_pred_labels.append(pred_labels)
    
    print(f"  ✓ 預測完成")

print("\n✓ 所有模型預測完成！")

# 7. Ensemble 方法 1：軟投票（機率平均）
print("\n" + "="*70)
print("📊 Ensemble 方法 1：軟投票（Soft Voting - 機率平均）")
print("="*70)

# 平均所有模型的預測機率
avg_predictions = np.mean(all_predictions, axis=0)
soft_voting_labels = np.argmax(avg_predictions, axis=1)

# 產生提交檔案
submission_soft = pd.DataFrame({
    'ImageId': range(1, len(soft_voting_labels) + 1),
    'Label': soft_voting_labels
})
submission_soft.to_csv('submission_ensemble_soft_voting.csv', index=False)

print("✓ 軟投票預測完成")
print(f"✓ 已儲存：submission_ensemble_soft_voting.csv")
print(f"\n預測標籤分佈：")
print(submission_soft['Label'].value_counts().sort_index())

# 8. Ensemble 方法 2：硬投票（多數決）
print("\n" + "="*70)
print("📊 Ensemble 方法 2：硬投票（Hard Voting - 多數決）")
print("="*70)

# 對每個測試樣本，統計各模型的預測結果，取最多的
hard_voting_labels = []
for i in range(len(X_test)):
    votes = [pred_labels[i] for pred_labels in all_pred_labels]
    # 統計投票，取最多的
    most_common = Counter(votes).most_common(1)[0][0]
    hard_voting_labels.append(most_common)

hard_voting_labels = np.array(hard_voting_labels)

# 產生提交檔案
submission_hard = pd.DataFrame({
    'ImageId': range(1, len(hard_voting_labels) + 1),
    'Label': hard_voting_labels
})
submission_hard.to_csv('submission_ensemble_hard_voting.csv', index=False)

print("✓ 硬投票預測完成")
print(f"✓ 已儲存：submission_ensemble_hard_voting.csv")
print(f"\n預測標籤分佈：")
print(submission_hard['Label'].value_counts().sort_index())

# 9. Ensemble 方法 3：加權軟投票（根據驗證集準確率加權）
print("\n" + "="*70)
print("📊 Ensemble 方法 3：加權軟投票（根據驗證集準確率）")
print("="*70)

# 計算權重（驗證集準確率）
weights = np.array([model_dict['val_accuracy'] for model_dict in models])
weights = weights / weights.sum()  # 正規化

print("模型權重：")
for i, (model_dict, weight) in enumerate(zip(models, weights)):
    print(f"  {model_dict['name']}: {weight:.4f} (驗證集準確率: {model_dict['val_accuracy']:.4f})")

# 加權平均
weighted_predictions = np.average(all_predictions, axis=0, weights=weights)
weighted_voting_labels = np.argmax(weighted_predictions, axis=1)

# 產生提交檔案
submission_weighted = pd.DataFrame({
    'ImageId': range(1, len(weighted_voting_labels) + 1),
    'Label': weighted_voting_labels
})
submission_weighted.to_csv('submission_ensemble_weighted_voting.csv', index=False)

print("\n✓ 加權軟投票預測完成")
print(f"✓ 已儲存：submission_ensemble_weighted_voting.csv")
print(f"\n預測標籤分佈：")
print(submission_weighted['Label'].value_counts().sort_index())

# 10. 比較三種方法的差異
print("\n" + "="*70)
print("🔍 比較三種 Ensemble 方法")
print("="*70)

# 計算三種方法的一致性
soft_vs_hard = np.sum(soft_voting_labels == hard_voting_labels)
soft_vs_weighted = np.sum(soft_voting_labels == weighted_voting_labels)
hard_vs_weighted = np.sum(hard_voting_labels == weighted_voting_labels)

total = len(X_test)
print(f"\n一致性分析：")
print(f"  軟投票 vs 硬投票：{soft_vs_hard}/{total} ({soft_vs_hard/total*100:.2f}%)")
print(f"  軟投票 vs 加權投票：{soft_vs_weighted}/{total} ({soft_vs_weighted/total*100:.2f}%)")
print(f"  硬投票 vs 加權投票：{hard_vs_weighted}/{total} ({hard_vs_weighted/total*100:.2f}%)")

# 找出三種方法不一致的樣本
disagreement = (soft_voting_labels != hard_voting_labels) | (soft_voting_labels != weighted_voting_labels)
n_disagreement = np.sum(disagreement)
print(f"\n三種方法有分歧的樣本數：{n_disagreement} ({n_disagreement/total*100:.2f}%)")

# 11. 最終總結
print("\n" + "="*70)
print("✅ Ensemble 預測完成！")
print("="*70)

print("\n產生的檔案：")
print("  📄 submission_ensemble_soft_voting.csv     ← 軟投票（推薦）")
print("  📄 submission_ensemble_hard_voting.csv     ← 硬投票")
print("  📄 submission_ensemble_weighted_voting.csv ← 加權軟投票（推薦）")

print("\n💡 建議提交順序：")
print("  1. 先提交：submission_ensemble_weighted_voting.csv")
print("     （加權軟投票通常效果最好）")
print("  2. 如果效果不理想，再試：submission_ensemble_soft_voting.csv")
print("  3. 最後試：submission_ensemble_hard_voting.csv")

print("\n📊 預期效果：")
print("  - Ensemble 通常比單一模型提升 0.2-0.5%")
print("  - 如果單一模型最好是 0.995，Ensemble 可能達到 0.997-0.998")

print("="*70)
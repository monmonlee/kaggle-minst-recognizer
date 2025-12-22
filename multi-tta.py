import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

print("="*70)
print("🎯 多種 TTA 策略組合")
print("="*70)

# 1. 載入測試資料
print("\n📂 載入測試資料...")
X_test = np.load('X_test.npy')
print(f"✓ 測試集載入完成：{X_test.shape}")

# 2. 載入模型
print("\n💾 載入模型...")
with open('ensemble_models_info.json', 'r') as f:
    models_info = json.load(f)

models = []
for info in models_info:
    model_path = f"{info['model_name']}_best.keras"
    try:
        model = load_model(model_path)
        models.append(model)
        print(f"  ✓ 載入：{info['model_name']}")
    except:
        pass

print(f"\n✓ 成功載入 {len(models)} 個模型")

# 3. 建立 TTA 生成器
print("\n🔄 建立 TTA 生成器...")
tta_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# 4. TTA 預測函數
def predict_with_tta(model, X, n_augmentations):
    predictions = model.predict(X, verbose=0)
    for i in range(n_augmentations - 1):
        aug_generator = tta_datagen.flow(X, batch_size=len(X), shuffle=False)
        X_aug = next(aug_generator)
        predictions += model.predict(X_aug, verbose=0)
    return predictions / n_augmentations

# 5. 策略：不同模型用不同 TTA 次數
print("\n" + "="*70)
print("🎯 策略：為每個模型找最佳 TTA 次數")
print("="*70)

# 這個策略是：讓不同模型用不同的 TTA 次數
# 因為每個模型可能有自己的最佳點

print("\n為每個模型測試不同 TTA 次數（這會需要一些時間）...")

# 定義要測試的 TTA 次數
tta_options = [10, 12, 15, 18, 20]

# 對每個模型，用不同 TTA 次數預測
all_model_predictions = []

for model_idx, model in enumerate(models, 1):
    print(f"\n【模型 {model_idx}/{len(models)}】")
    model_predictions_by_tta = {}
    
    for n_tta in tta_options:
        print(f"  TTA x{n_tta}...", end=' ')
        preds = predict_with_tta(model, X_test, n_tta)
        model_predictions_by_tta[n_tta] = preds
        print("✓")
    
    all_model_predictions.append(model_predictions_by_tta)

print("\n✓ 所有預測完成")

# 6. 組合策略
print("\n" + "="*70)
print("📊 測試不同的組合策略")
print("="*70)

strategies = {}

# 策略 A：所有模型都用 TTA x15
print("\n策略 A：所有模型統一用 TTA x15")
preds_a = []
for model_preds in all_model_predictions:
    preds_a.append(model_preds[15])
avg_preds_a = np.mean(preds_a, axis=0)
labels_a = np.argmax(avg_preds_a, axis=1)
strategies['A_TTA15統一'] = labels_a

# 策略 B：所有模型都用 TTA x20
print("策略 B：所有模型統一用 TTA x20")
preds_b = []
for model_preds in all_model_predictions:
    preds_b.append(model_preds[20])
avg_preds_b = np.mean(preds_b, axis=0)
labels_b = np.argmax(avg_preds_b, axis=1)
strategies['B_TTA20統一'] = labels_b

# 策略 C：混合不同 TTA（每個模型用不同的）
print("策略 C：每個模型用不同 TTA 次數")
tta_assignments = [15, 20, 18, 12, 15]  # 為每個模型分配不同 TTA
preds_c = []
for i, model_preds in enumerate(all_model_predictions):
    assigned_tta = tta_assignments[i] if i < len(tta_assignments) else 15
    preds_c.append(model_preds[assigned_tta])
avg_preds_c = np.mean(preds_c, axis=0)
labels_c = np.argmax(avg_preds_c, axis=1)
strategies['C_TTA混合'] = labels_c

# 策略 D：對每個模型，平均多個 TTA 的結果
print("策略 D：每個模型平均多個 TTA 結果")
preds_d = []
for model_preds in all_model_predictions:
    # 平均該模型所有 TTA 的預測
    avg_model_pred = np.mean([model_preds[tta] for tta in tta_options], axis=0)
    preds_d.append(avg_model_pred)
avg_preds_d = np.mean(preds_d, axis=0)
labels_d = np.argmax(avg_preds_d, axis=1)
strategies['D_TTA多重平均'] = labels_d

# 策略 E：超級 Ensemble（所有模型 × 所有 TTA）
print("策略 E：超級 Ensemble（所有組合）")
all_predictions = []
for model_preds in all_model_predictions:
    for tta in tta_options:
        all_predictions.append(model_preds[tta])
avg_preds_e = np.mean(all_predictions, axis=0)
labels_e = np.argmax(avg_preds_e, axis=1)
strategies['E_超級Ensemble'] = labels_e

# 7. 產生所有策略的提交檔案
print("\n" + "="*70)
print("📄 產生提交檔案")
print("="*70)

for strategy_name, labels in strategies.items():
    submission = pd.DataFrame({
        'ImageId': range(1, len(labels) + 1),
        'Label': labels
    })
    
    filename = f"submission_tta_{strategy_name}.csv"
    submission.to_csv(filename, index=False)
    print(f"✓ 已儲存：{filename}")
    print(f"  標籤分佈：{pd.Series(labels).value_counts().sort_index().to_dict()}")

# 8. 分析策略之間的差異
print("\n" + "="*70)
print("🔍 分析不同策略的差異")
print("="*70)

strategy_names = list(strategies.keys())
for i in range(len(strategy_names)):
    for j in range(i+1, len(strategy_names)):
        name1, name2 = strategy_names[i], strategy_names[j]
        labels1, labels2 = strategies[name1], strategies[name2]
        
        agreement = np.sum(labels1 == labels2)
        disagreement = len(labels1) - agreement
        
        print(f"\n{name1} vs {name2}:")
        print(f"  一致：{agreement} / {len(labels1)} ({agreement/len(labels1)*100:.2f}%)")
        print(f"  不同：{disagreement} ({disagreement/len(labels1)*100:.2f}%)")

# 9. 總結
print("\n" + "="*70)
print("✅ 多重 TTA 策略完成！")
print("="*70)

print("\n產生的檔案：")
for strategy_name in strategies.keys():
    print(f"  📄 submission_tta_{strategy_name}.csv")

print("\n💡 建議提交順序：")
print("  1. submission_tta_E_超級Ensemble.csv  ← 最推薦（組合最多）")
print("  2. submission_tta_D_TTA多重平均.csv")
print("  3. submission_tta_C_TTA混合.csv")
print("  4. submission_tta_B_TTA20統一.csv")

print("\n📊 預期效果：")
print("  - 超級 Ensemble 組合了 25 個預測（5 模型 × 5 TTA）")
print("  - 理論上應該最穩定")
print("  - 預期提升：0.0005-0.001")

print("="*70)
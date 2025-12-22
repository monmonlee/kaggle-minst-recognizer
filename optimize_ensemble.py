import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from itertools import product
import pandas as pd

print("="*70)
print("🎯 優化 Ensemble 權重 - 在驗證集上搜尋最佳組合")
print("="*70)

# 1. 載入驗證集資料
print("\n📂 載入驗證集資料...")
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
print(f"✓ 驗證集載入完成：{X_val.shape}")

# 2. 載入測試集資料
print("\n📂 載入測試集資料...")
X_test = np.load('X_test.npy')
print(f"✓ 測試集載入完成：{X_test.shape}")

# 3. 載入模型資訊
print("\n📂 載入模型資訊...")
with open('ensemble_models_info.json', 'r') as f:
    models_info = json.load(f)

print(f"✓ 找到 {len(models_info)} 個模型")

# 4. 載入所有模型
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
        print(f"  ✓ 載入：{info['model_name']} (驗證集準確率: {info['val_accuracy']:.4f})")
    except Exception as e:
        print(f"  ⚠️  無法載入 {model_path}")

n_models = len(models)
print(f"\n✓ 成功載入 {n_models} 個模型")

# 5. 對驗證集進行預測（不用 TTA，更快）
print("\n🔮 對驗證集進行預測...")
val_predictions = []
for i, model_dict in enumerate(models, 1):
    model = model_dict['model']
    print(f"  [{i}/{n_models}] {model_dict['name']}")
    preds = model.predict(X_val, verbose=0)
    val_predictions.append(preds)

print("✓ 驗證集預測完成")

# 6. 搜尋最佳權重組合
print("\n" + "="*70)
print("🔍 在驗證集上搜尋最佳權重組合")
print("="*70)

def weighted_ensemble(predictions, weights):
    """使用權重組合預測"""
    weighted_preds = np.average(predictions, axis=0, weights=weights)
    return np.argmax(weighted_preds, axis=1)

# 方法 1：網格搜尋（粗略）
print("\n【方法 1】粗略網格搜尋...")
print("範圍：每個模型權重 0.1 - 0.3（步長 0.05）")

best_accuracy = 0
best_weights = None
search_count = 0

# 生成權重候選（確保總和為 1）
weight_options = np.arange(0.1, 0.35, 0.05)

# 限制搜尋空間（避免太久）
# 只搜尋前 1000 種組合
max_searches = 1000
search_step = 0

for weights_combo in product(weight_options, repeat=n_models):
    search_step += 1
    if search_step > max_searches:
        break
    
    weights = np.array(weights_combo)
    
    # 正規化權重（總和為 1）
    if weights.sum() == 0:
        continue
    weights = weights / weights.sum()
    
    # 組合預測
    ensemble_preds = weighted_ensemble(val_predictions, weights)
    accuracy = accuracy_score(y_val, ensemble_preds)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = weights
        print(f"  🎉 找到更好的組合！準確率：{accuracy:.5f}")
        print(f"     權重：{weights}")
    
    if search_step % 100 == 0:
        print(f"  已搜尋：{search_step}/{max_searches}")

print(f"\n✓ 粗略搜尋完成")
print(f"✓ 最佳驗證集準確率：{best_accuracy:.5f}")
print(f"✓ 最佳權重：{best_weights}")

# 7. 方法 2：基於驗證集準確率的智能權重
print("\n" + "="*70)
print("【方法 2】基於驗證集準確率的進階權重")
print("="*70)

# 取得每個模型的驗證集準確率
val_accuracies = np.array([m['val_accuracy'] for m in models])

# 策略 A：線性權重
linear_weights = val_accuracies / val_accuracies.sum()
linear_preds = weighted_ensemble(val_predictions, linear_weights)
linear_accuracy = accuracy_score(y_val, linear_preds)
print(f"\n策略 A（線性權重）：")
print(f"  權重：{linear_weights}")
print(f"  準確率：{linear_accuracy:.5f}")

# 策略 B：平方權重（強化好模型）
squared_weights = val_accuracies ** 2
squared_weights = squared_weights / squared_weights.sum()
squared_preds = weighted_ensemble(val_predictions, squared_weights)
squared_accuracy = accuracy_score(y_val, squared_preds)
print(f"\n策略 B（平方權重）：")
print(f"  權重：{squared_weights}")
print(f"  準確率：{squared_accuracy:.5f}")

# 策略 C：指數權重（更強化好模型）
exp_weights = np.exp(val_accuracies * 10)  # 放大差異
exp_weights = exp_weights / exp_weights.sum()
exp_preds = weighted_ensemble(val_predictions, exp_weights)
exp_accuracy = accuracy_score(y_val, exp_preds)
print(f"\n策略 C（指數權重）：")
print(f"  權重：{exp_weights}")
print(f"  準確率：{exp_accuracy:.5f}")

# 策略 D：只用最好的 3 個模型
top3_indices = np.argsort(val_accuracies)[-3:]
top3_weights = np.zeros(n_models)
top3_weights[top3_indices] = val_accuracies[top3_indices]
top3_weights = top3_weights / top3_weights.sum()
top3_preds = weighted_ensemble(val_predictions, top3_weights)
top3_accuracy = accuracy_score(y_val, top3_preds)
print(f"\n策略 D（只用前 3 名）：")
print(f"  權重：{top3_weights}")
print(f"  準確率：{top3_accuracy:.5f}")

# 8. 比較所有策略
print("\n" + "="*70)
print("📊 所有策略比較")
print("="*70)

strategies = {
    '網格搜尋最佳': (best_weights, best_accuracy),
    '線性權重': (linear_weights, linear_accuracy),
    '平方權重': (squared_weights, squared_accuracy),
    '指數權重': (exp_weights, exp_accuracy),
    '只用前3名': (top3_weights, top3_accuracy)
}

# 排序
sorted_strategies = sorted(strategies.items(), key=lambda x: x[1][1], reverse=True)

print("\n驗證集準確率排名：")
for i, (name, (weights, acc)) in enumerate(sorted_strategies, 1):
    print(f"{i}. {name:15s}: {acc:.5f}")

# 找出最佳策略
best_strategy_name, (final_best_weights, final_best_accuracy) = sorted_strategies[0]

print(f"\n🏆 最佳策略：{best_strategy_name}")
print(f"   驗證集準確率：{final_best_accuracy:.5f}")
print(f"   權重：{final_best_weights}")

# 9. 使用最佳權重預測測試集
print("\n" + "="*70)
print("🚀 使用最佳權重預測測試集")
print("="*70)

# 對測試集進行預測
print("\n對測試集進行預測...")
test_predictions = []
for i, model_dict in enumerate(models, 1):
    model = model_dict['model']
    print(f"  [{i}/{n_models}] {model_dict['name']}")
    preds = model.predict(X_test, verbose=0)
    test_predictions.append(preds)

# 使用最佳權重組合
final_test_preds = weighted_ensemble(test_predictions, final_best_weights)

# 產生提交檔案
submission = pd.DataFrame({
    'ImageId': range(1, len(final_test_preds) + 1),
    'Label': final_test_preds
})

output_filename = f'submission_optimized_weights.csv'
submission.to_csv(output_filename, index=False)

print(f"\n✓ 測試集預測完成")
print(f"✓ 已儲存：{output_filename}")
print(f"\n預測標籤分佈：")
print(submission['Label'].value_counts().sort_index())

# 10. 同時產生所有策略的提交檔案
print("\n" + "="*70)
print("📄 產生所有策略的提交檔案")
print("="*70)

for strategy_name, (weights, val_acc) in strategies.items():
    test_preds = weighted_ensemble(test_predictions, weights)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(test_preds) + 1),
        'Label': test_preds
    })
    
    filename = f"submission_{strategy_name.replace(' ', '_')}.csv"
    submission.to_csv(filename, index=False)
    print(f"✓ 已儲存：{filename} (驗證集準確率: {val_acc:.5f})")

# 11. 總結
print("\n" + "="*70)
print("✅ 權重優化完成！")
print("="*70)

print("\n💡 建議提交順序：")
for i, (name, (_, acc)) in enumerate(sorted_strategies, 1):
    filename = f"submission_{name.replace(' ', '_')}.csv"
    print(f"{i}. {filename}")
    print(f"   (驗證集準確率: {acc:.5f})")

print("\n📊 預期效果：")
print(f"  - 最佳策略驗證集準確率：{final_best_accuracy:.5f}")
print(f"  - 相比平均權重可能提升：0.0001-0.0005")
print(f"  - 測試集預期分數：0.996-0.997")

print("="*70)
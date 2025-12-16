import tensorflow as tf
import sys

print("="*60)
print("🔍 TensorFlow GPU 檢測")
print("="*60)

# 1. TensorFlow 版本
print(f"\nTensorFlow 版本：{tf.__version__}")
print(f"Python 版本：{sys.version}")

# 2. 檢查是否有 GPU
print("\n📊 可用的裝置：")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"  - {device.device_type}: {device.name}")

# 3. 檢查 GPU 裝置
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🎮 GPU 數量：{len(gpus)}")

if gpus:
    print("✅ 找到 GPU！")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("❌ 沒有找到 GPU（可能使用 CPU）")

# 4. 測試 GPU 計算
print("\n⚡ 測試 GPU 計算速度...")
import time

# CPU 測試
with tf.device('/CPU:0'):
    start = time.time()
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"  CPU 計算時間：{cpu_time:.4f} 秒")

# GPU 測試（如果有的話）
if gpus:
    with tf.device('/GPU:0'):
        start = time.time()
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
        gpu_time = time.time() - start
        print(f"  GPU 計算時間：{gpu_time:.4f} 秒")
        print(f"  加速比：{cpu_time/gpu_time:.2f}x")

print("\n" + "="*60)
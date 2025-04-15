import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, toeplitz

# 配置 Matplotlib 以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def generate_ar1_sequence(rho, N):
    """
    生成 AR(1) 序列 x[n] = rho*x[n-1] + w[n]
    其中 w[n] 是白噪声，其选择方式使得输出序列的方差为 1，
    自相关函数为 rho**|k|。
    sigma=1 由此保证
    返回生成的序列 x 和使用的白噪声 w。
    """
    if not (-1 < rho < 1):
        raise ValueError("rho 必须在 -1 和 1 之间以保证平稳性")

    # 为了使输出方差为 1 所需的白噪声输入的方差
    sigma_w2 = 1 - rho**2
    # 生成高斯白噪声 w
    w = np.random.normal(0, np.sqrt(sigma_w2), N)

    x = np.zeros(N)
    # 初始化第一个值
    x[0] = w[0]

    for n in range(1, N):
        x[n] = rho * x[n-1] + w[n]
    # 返回生成的序列和内部使用的白噪声
    return x, w, sigma_w2

def calculate_sample_correlation(x, M):
    """
    计算样本自相关函数 R_hat(k)，滞后范围从 0 到 M。
    使用有偏估计量： R_hat(k) = (1/N) * sum(x[n]*x[n-k])
    """
    N = len(x)
    if M >= N:
        raise ValueError("最大滞后 M 必须小于序列长度 N")

    sample_R = np.zeros(M + 1)
    # 使用定义 R(k) = E[x(n)x(n-k)]
    # 通过样本均值 (1/N) * sum(...) 来估计 E[...]

    sample_R[0] = np.sum(x**2) / N
    for k in range(1, M + 1):
         # 滞后 k 的相关性：对 n=k 到 N-1 求和 x[n]*x[n-k]
         # 等价于对 x[k:] * x[:-k] 求和
        sample_R[k] = np.sum(x[k:] * x[:-k]) / N

    return sample_R

def generate_sequence_cholesky(rho, N):
    """
    使用 Cholesky 分解生成自相关为 rho**|k| 的序列。
    """
    # 创建 N x N 的理论协方差矩阵 Sigma
    # Sigma[i, j] = rho**|i-j|
    k = np.arange(N)
    cov_matrix = rho**np.abs(k[:, None] - k[None, :])

    # 执行 Cholesky 分解： Sigma = L * L.T
    try:
        L = cholesky(cov_matrix, lower=True)
    except np.linalg.LinAlgError:
        print("协方差矩阵不是正定的。")
        # 可能由于数值问题发生，特别是当 rho 接近 1 时
        # 添加小的对角线噪声（nugget）
        epsilon = 1e-9
        cov_matrix += np.eye(N) * epsilon
        L = cholesky(cov_matrix, lower=True)

    # 生成白噪声 w ~ N(0, I)
    w_chol = np.random.normal(0, 1, N)

    # 生成相关序列 x = L * w
    x_chol = L @ w_chol
    return x_chol


# --- 参数 ---
rho = 0.9  # 相关系数
N = 1000   # 随机序列的长度
M = 50     # 相关计算的最大滞后

# --- 1. 设计滤波器 & 3. 通过滤波生成序列 ---
print("使用 AR(1) 滤波方法生成序列...")
# x_filter是输出序列，w_filter是对应的输入白噪声
x_filter, w_filter, sigma_w2 = generate_ar1_sequence(rho, N)

# --- 理论相关函数 ---
lags = np.arange(M + 1)
theoretical_R_x = rho**lags # 输出序列 x 的理论自相关 R_x(k) = sigma_x^2 * rho^|k| = 1 * rho^|k|

# 理论上的输入白噪声自相关 R_w(k)
theoretical_R_w = np.zeros(M + 1)
theoretical_R_w[0] = sigma_w2 # R_w(0) = sigma_w^2, R_w(k) = 0 for k != 0

# --- 4. 计算样本相关性和均方误差 ---
print(f"计算输出序列 x 的样本相关性 (滞后 M={M})...")
sample_R_x_filter = calculate_sample_correlation(x_filter, M)
mse_x = np.mean((sample_R_x_filter - theoretical_R_x)**2)
print(f"输出序列 x：样本与理论相关性的 MSE: {mse_x:.6f}")

print(f"计算输入噪声 w 的样本相关性 (滞后 M={M})...")
sample_R_w_filter = calculate_sample_correlation(w_filter, M)
mse_w = np.mean((sample_R_w_filter - theoretical_R_w)**2)
print(f"输入噪声 w：样本与理论相关性的 MSE: {mse_w:.6f}")


# --- 可视化 ---
plt.figure(figsize=(12, 10))

# 绘制生成的序列（前100个点）
plt.subplot(3, 1, 1)
plt.plot(x_filter[:100])
plt.title(f'生成的 AR(1) 序列 x (滤波方法, 前 100 点), rho={rho}')
plt.xlabel('样本点 n')
plt.ylabel('序列值 x[n]')
plt.grid(True)

# 绘制理论与样本相关性 (输出序列 x)
plt.subplot(3, 1, 2)
plt.plot(lags, theoretical_R_x, 'bo-', label='理论 R_x(k)')
plt.plot(lags, sample_R_x_filter, 'rx--', label='样本 R_x(k) (滤波)')
plt.title('输出序列 x 的理论 vs. 样本自相关函数')
plt.xlabel('滞后 k')
plt.ylabel('R_x(k)')
plt.legend()
plt.grid(True)

# 绘制理论与样本相关性 (输入噪声 w)
plt.subplot(3, 1, 3)
plt.plot(lags, theoretical_R_w, 'go-', label='理论 R_w(k)')
plt.plot(lags, sample_R_w_filter, 'mx--', label='样本 R_w(k) (滤波)')
plt.title('输入白噪声 w 的理论 vs. 样本自相关函数')
plt.xlabel('滞后 k')
plt.ylabel('R_w(k)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- 5. 分析与 Cholesky 方法的差异 ---
print("\n使用 Cholesky 方法生成序列以进行比较...")
if N <= 2000:
    x_chol = generate_sequence_cholesky(rho, N)
    sample_R_x_chol = calculate_sample_correlation(x_chol, M)
    mse_x_chol = np.mean((sample_R_x_chol - theoretical_R_x)**2)
    print(f"Cholesky 方法 (输出序列 x) 的 MSE: {mse_x_chol:.6f}")

    # 绘制 Cholesky 方法与滤波方法输出序列 x 的自相关比较图
    plt.figure(figsize=(10, 5))
    plt.plot(lags, theoretical_R_x, 'bo-', label='理论 R_x(k)')
    plt.plot(lags, sample_R_x_filter, 'rx--', label=f'样本 R_x(k) (滤波, MSE={mse_x:.4f})')
    plt.plot(lags, sample_R_x_chol, 'g+--', label=f'样本 R_x(k) (Cholesky, MSE={mse_x_chol:.4f})')
    plt.title('输出序列 x 自相关比较: 滤波 vs. Cholesky')
    plt.xlabel('滞后 k')
    plt.ylabel('R_x(k)')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print(f"对于大的 N={N}，跳过 Cholesky 方法的生成。")


print("\n滤波方法与 Cholesky 方法的比较:")
print("1. 滤波 (AR(1) 模拟):")
print("   - 直接模拟底层过程 (x[n] = rho*x[n-1] + w[n])。")
print("   - 计算效率高：O(N) 复杂度。")
print("   - 迭代生成序列。")
print("   - 结果是平稳过程的近似；可能需要预热（warm-up）。")
print("   - 对于大的 N，样本相关性收敛于理论值。")
print("2. Cholesky 分解:")
print("   - 直接强制施加所需的协方差结构 (Sigma = L*L.T)。")
print("   - 计算成本高：分解为 O(N^3)，生成为 O(N^2)。")
print("   - 需要存储 N x N 协方差矩阵 (O(N^2) 内存)。")
print("   - 同时生成整个序列。")
print("   - 对于生成的有限序列，其相关结构在数值上是精确的（在机器精度内）。")
print('   - 通常被认为是生成具有特定协方差的多元正态分布的"黄金标准"，但对于非常长的时间序列，不如滤波方法实用。')

# --- 6. 其他生成方法 ---
print("\n生成相关序列的其他方法:")
print("- 基于 FFT 的方法：使用功率谱密度 (PSD)，对于大的 N 高效 (O(N log N))。例如 Davies-Harte 算法。")
print("- 状态空间模型 / 卡尔曼滤波：通用框架，对于 ARMA 过程等效于直接滤波。")
print("- 移动平均 (MA) 滤波器：另一种通过滤波白噪声生成相关噪声的常用方法。")
print("- 直接多元正态采样：使用像 `numpy.random.multivariate_normal` 这样的库，内部通常使用 Cholesky 或 SVD。")

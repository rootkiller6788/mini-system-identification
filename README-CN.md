# Mini 系统辨识

一套**从零构建、零依赖的 C 语言实现**，覆盖系统辨识理论——从观测到的输入-输出数据中建立动态系统的数学模型。每个子模块对应 MIT 和 Stanford 的课程体系，将教材中的数学方程转化为可运行的 C 代码，架起理论与实践之间的桥梁。

## 子模块

| 子模块 | 主题 | 对应课程 |
|--------|------|----------|
| [mini-closed-loop-identification](mini-closed-loop-identification/) | 直接/间接闭环辨识，工具变量法（IV），联合输入-输出辨识，子空间闭环方法，Youla 参数化，模型验证 | MIT 6.241J, Ljung (1999) Ch.13 |
| [mini-frequency-domain-id](mini-frequency-domain-id/) | 频率响应估计（ETFE、H1/H2/相干系数），频谱/倒谱分析，扫频/多正弦激励，s-z 域转换，谐振检测，描述函数 | MIT 6.241J, Pintelon & Schoukens (2012) |
| [mini-nonlinear-system-id](mini-nonlinear-system-id/) | NARX、NARMAX、Wiener/Hammerstein/Wiener-Hammerstein 模型，Gauss-Newton、Levenberg-Marquardt、正交最小二乘、递推最小二乘、交叉验证 | MIT 6.241J, Nelles (2001) |
| [mini-prediction-error-method](mini-prediction-error-method/) | ARX、ARMAX、OE、BJ 模型结构，二次/Huber/Vapnik 损失函数，Newton/Gauss-Newton 优化，AIC/BIC 模型选择，残差分析 | MIT 6.241J, Ljung (1999), Söderström & Stoica (1989) |
| [mini-regularized-least-squares](mini-regularized-least-squares/) | 岭回归（L2）、Lasso（L1）、弹性网络，核方法（Stable Spline、TC），Cholesky/QR 求解器，偏差-方差权衡，超参数调优 | MIT 6.241J, Stanford EE364A, Pillonetto et al. (2014) |
| [mini-subspace-identification](mini-subspace-identification/) | N4SID、MOESP、CVA，Hankel 矩阵，斜投影，基于 SVD 的阶次选择，LQ 分解，工具变量子空间 | MIT 6.241J, Van Overschee & De Moor (1996), Katayama (2005) |
| [mini-uncertainty-quantification](mini-uncertainty-quantification/) | 贝叶斯推断，MCMC（Metropolis-Hastings/Gibbs），多项式混沌展开，Sobol 灵敏度指标，可信/置信区域，参数不确定性传播 | Stanford CME 206, Smith (2013), Sullivan (2015) |
| [mini-wiener-hammerstein](mini-wiener-hammerstein/) | FIR/IIR/SS 线性块，静态非线性（死区/饱和/迟滞），最佳线性近似（BLA），迭代辨识，过参数化，非线性失真分析 | MIT 6.241J, Schoukens et al. (2005) |

## 设计理念

- **零外部依赖** — 纯 C（C99/C11），仅使用 `libc` 和 `libm`
- **模块独立** — 每个目录包含自身的 `Makefile`、`include/`、`src/`、`examples/`、`demos/`、`tests/`、`docs/`
- **理论到代码的映射** — 每个模块实现了经典教材（Ljung、Söderström & Stoica、Van Overschee & De Moor）中的核心算法
- **实用演示** — 直流电机辨识、四旋翼无人机、化工过程、GPS 去噪、生物医学血糖监测、颤振分析等

## 构建方式

每个模块独立构建。进入模块目录后运行：

```bash
cd mini-closed-loop-identification
make all    # 构建全部
make test   # 运行测试
```

需要 **GCC** 和 **GNU Make**。

## 目录结构

```
mini-system-identification/
├── mini-closed-loop-identification/   # 直接/间接闭环辨识、工具变量法、联合输入-输出、Youla 参数化
├── mini-frequency-domain-id/          # 频率响应估计、频谱分析、扫频/多正弦激励、描述函数
├── mini-nonlinear-system-id/          # NARX/NARMAX、Wiener/Hammerstein 模型、Gauss-Newton、LM、递推最小二乘
├── mini-prediction-error-method/      # ARX/ARMAX/OE/BJ、损失函数、Newton/Gauss-Newton、AIC/BIC
├── mini-regularized-least-squares/    # 岭回归/Lasso/弹性网络、核方法（SS、TC）、偏差-方差权衡
├── mini-subspace-identification/      # N4SID/MOESP/CVA、Hankel 矩阵、斜投影、SVD 阶次选择
├── mini-uncertainty-quantification/   # 贝叶斯不确定性量化、MCMC、多项式混沌、Sobol 灵敏度、可信区域
└── mini-wiener-hammerstein/           # FIR/IIR/SS 线性块、非线性特性（死区/饱和/迟滞）、BLA
```

## 许可证

MIT

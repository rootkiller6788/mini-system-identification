# mini-regularized-least-squares

**模块**: 10. mini-system-identification
**主题**: Regularized Least Squares for System Identification (正则化最小二乘法系统辨识)

## Module Status: COMPLETE

- **L1-L6**: Complete
- **L7**: Complete (4 applications: DC motor, FOPDT process control, GPS denoising, biomedical glucose)
- **L8**: Complete (kernel-based regularization, stable spline, TC kernel, Empirical Bayes)
- **L9**: Partial (documented, research frontiers in kernel learning)

---

## 九层知识覆盖摘要

| Level | 名称 | 状态 | 关键条目 |
|-------|------|------|---------|
| **L1** | Definitions | Complete | 7 struct definitions, 6 enums, regularized least squares, model types |
| **L2** | Core Concepts | Complete | Bias-variance tradeoff, overfitting, cross-validation, condition number, effective df |
| **L3** | Math Structures | Complete | Column-major matrix, SVD, QR, Cholesky, LDL^T, eigenvalue, pseudo-inverse |
| **L4** | Fundamental Laws | Complete | Normal equations theorem, Gauss-Markov, ridge bias formula, SVD shrinkage |
| **L5** | Algorithms | Complete | Ridge (Cholesky/SVD/QR/CG), LASSO (CD/ADMM), Elastic Net, Group LASSO, Fused LASSO, LSQR |
| **L6** | Canonical Problems | Complete | FIR identification, ARX estimation, ill-conditioned regression, sparse recovery |
| **L7** | Applications | Complete | DC motor (Ljung), FOPDT process, GPS signal denoising, biomedical glucose |
| **L8** | Advanced Topics | Complete | Kernel-based regularization, Stable Spline, TC/DI/DC kernels, marginal likelihood, hyperparameter optimization |
| **L9** | Research Frontiers | Partial | Deep kernel learning, sparse Bayesian learning (documented) |

---

## 核心定义 (L1)

- **Regularized Least Squares**: `min_theta ||y - Phi*theta||^2 + P(theta)`
- **Ridge (L2/Tikhonov)**: `P(theta) = lambda * ||theta||_2^2`
- **LASSO (L1)**: `P(theta) = lambda * ||theta||_1`
- **Elastic Net**: `P(theta) = lambda*(alpha*||theta||_1 + (1-alpha)*||theta||_2^2/2)`
- **ARX Model**: `A(q)*y(t) = B(q)*u(t) + e(t)`
- **FIR Model**: `y(t) = sum_j b_j*u(t-j) + e(t)`
- **Kernel Regularization**: `theta ~ N(0, lambda*K)` with kernel matrix K

---

## 核心定理 (L4)

1. **Normal Equations** (Gauss, 1821): `(Phi^T*Phi + lambda*I) * theta = Phi^T*y` uniquely minimizes the ridge objective for lambda>0 (strong convexity).

2. **Gauss-Markov Theorem** (extended): Among linear unbiased estimators with regularization, the ridge estimator minimizes MSE under the bias-variance tradeoff.

3. **Ridge Bias**: `Bias(theta_hat) = -lambda * (Phi^T*Phi + lambda*I)^(-1) * theta_true`

4. **SVD Shrinkage**: `theta_ridge = sum_i (s_i/(s_i^2+lambda)) * (u_i^T*y) * v_i` where s_i are singular values. Ridge shrinks small singular values.

5. **Effective Degrees of Freedom**: `df(lambda) = sum_i s_i^2/(s_i^2+lambda) = tr(Phi*(Phi^T*Phi+lambda*I)^(-1)*Phi^T)`

6. **GCV Theorem** (Golub-Heath-Wahba, 1979): GCV is a rotation-invariant approximately unbiased estimate of prediction risk.

7. **Representer Theorem** (Kimeldorf-Wahba, 1971): Kernel ridge solution is a finite linear combination of kernel evaluations at data points.

---

## 核心算法 (L5)

| 算法 | 复杂度 | 描述 |
|------|--------|------|
| Ridge-Cholesky | O(np^2 + p^3/3) | 直接法，适合 p ≤ 1000 |
| Ridge-SVD | O(np*min(n,p)) | 病态问题优选，提供奇异值谱 |
| Ridge-QR | O(2np^2) | 增广系统法，数值稳定 |
| Ridge-CG | O(k*nnz(Phi)) | 迭代法，适合大规模稀疏 |
| LASSO-CD | O(k*n*p) | 坐标下降，最常用的LASSO求解器 |
| LASSO-ADMM | O(k*(p^3+n*p)) | 交替方向乘子法 |
| ElasticNet-CD | O(k*n*p) | 弹性网坐标下降 |
| Group LASSO | O(k*n*p) | 块坐标下降 |
| Fused LASSO | O(k*(p^3+n*p)) | ADMM求解 |

---

## 经典问题 (L6)

1. **FIR系统辨识** — 从I/O数据估计脉冲响应 (example_fir_id.c)
2. **ARX模型估计** — 差分方程参数估计 (example_arx_id.c)
3. **病态回归** — 近共线设计矩阵的正则化 (example_ill_conditioned.c)
4. **稀疏恢复** — LASSO选择活跃回归量

---

## 九校课程映射

| 学校 | 课程 | 对标内容 |
|------|------|---------|
| **MIT** | 6.241J Dynamic Systems | State-space regression, Kalman |
| **Stanford** | EE363 Convex Optimization | Ridge/LASSO as convex programs |
| **Berkeley** | EE221A Linear Systems | Least squares, SVD system theory |
| **CMU** | 18-771 Linear Systems | Regularized estimation |
| **Princeton** | ELE 530 Estimation | Bias-variance, Stein estimation |
| **Caltech** | CDS110 Intro Ctrl | System identification basics |
| **Cambridge** | 4F3 Nonlinear Ctrl | Regularization in nonlinear ID |
| **Oxford** | C20 Adaptive Ctrl | Online regularization (RLS) |
| **ETH** | 227-0216 Sys Identification | Kernel methods, Ljung textbook |

---

## 构建与运行

```bash
make          # 构建静态库
make test     # 运行测试 (24 tests)
make examples # 构建示例
make demo     # 运行全部示例
make clean    # 清理构建产物
```

## 文件清单

```
include/  (6 headers)
  rls_core.h         核心类型、矩阵/向量代数、分解
  rls_regularizers.h 正则化惩罚函数、邻近算子
  rls_solvers.h      求解器 (Ridge/LASSO/ElasticNet/Group/Fused)
  rls_models.h       模型结构 (FIR/ARX/OE/ARMAX/BJ/SS/NARX)
  rls_validation.h   正则化参数选择 (GCV/L-curve/CV/AICc/SURE/EB)
  rls_kernel.h       核方法 (SS/TC/DI/DC/RBF/Matern)

src/  (8 implementations)
  rls_core.c         矩阵代数、Cholesky、SVD、QR、LDL^T、特征值
  rls_regularizers.c 惩罚评估、梯度/次梯度、邻近算子
  rls_solvers.c      Ridge/LASSO/ElasticNet/Group/Fused/LSQR求解器
  rls_models.c       回归量构造、模型仿真、验证指标
  rls_validation.c   K-fold CV、GCV、L-curve、AICc、SURE、Empirical Bayes
  rls_kernel.c       核矩阵、核岭回归、超参数优化
  rls_applications.c DC电机、FOPDT过程、GPS去噪、血糖动态 (L7)

tests/  test_rls.c   24个测试覆盖L1-L8
examples/  (3端到端示例)
  example_fir_id.c           FIR辨识 (Ridge/LASSO/Kernel/OLS对比)
  example_arx_id.c           ARX辨识 (lambda选择方法比较)
  example_ill_conditioned.c  病态回归 (正则化效果展示)
docs/  知识覆盖文档
```

---

## 参考教材

1. Ljung, L. "System Identification: Theory for the User" 2/e, 1999
2. Golub, G.H., Van Loan, C.F. "Matrix Computations" 3/e, 1996
3. Hastie, T., Tibshirani, R., Friedman, J. "Elements of Statistical Learning" 2/e, 2009
4. Pillonetto, G. et al. "Regularized System Identification", 2022
5. Hoerl, A.E., Kennard, R.W. "Ridge Regression", Technometrics 12(1), 1970
6. Tibshirani, R. "Regression Shrinkage via the Lasso", JRSS-B 58(1), 1996
7. Zou, H., Hastie, T. "Regularization via the Elastic Net", JRSS-B 67(2), 2005

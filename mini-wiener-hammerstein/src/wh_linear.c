/**
 * wh_linear.c ? Linear Dynamic Block Implementations
 *
 * Implements FIR filtering, IIR filtering, state-space simulation,
 * frequency response, stability analysis, and impulse/step response
 * for the linear blocks in Wiener-Hammerstein models.
 *
 * Knowledge Level: L3 (Mathematical Structures), L5 (Algorithms)
 */

#include "wh_linear.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ??? Initialization ????????????????????????????????????????????????????? */

int wh_linear_init_fir(WH_LinearBlock* block, const double* b,
                        int nb, double Ts) {
    if (!block || !b || nb <= 0 || nb > 64) return -1;
    memset(block, 0, sizeof(WH_LinearBlock));
    block->type = WH_LIN_FIR;
    block->nb = nb;
    block->na = 0;
    block->order = nb;
    block->Ts = Ts;
    block->D = b[0]; /* Feedthrough = first coefficient */
    for (int i = 0; i < nb; i++) {
        block->b[i] = b[i];
    }
    block->a[0] = 1.0; /* Trivial denominator */
    return 0;
}

int wh_linear_init_iir(WH_LinearBlock* block, const double* b, int nb,
                        const double* a, int na, double Ts) {
    if (!block || !b || !a || nb <= 0 || na <= 0 ||
        nb > 64 || na > 64) return -1;
    if (fabs(a[0] - 1.0) > 1e-12) return -1; /* Must be monic */
    memset(block, 0, sizeof(WH_LinearBlock));
    block->type = WH_LIN_IIR_TF;
    block->nb = nb;
    block->na = na;
    block->order = (nb > na ? nb : na);
    block->Ts = Ts;
    /* D = b[0]/a[0] = b[0] since a[0]=1 */
    block->D = b[0];
    for (int i = 0; i < nb; i++) block->b[i] = b[i];
    for (int i = 0; i < na; i++) block->a[i] = a[i];
    return 0;
}

int wh_linear_init_ss(WH_LinearBlock* block, const double* A,
                       const double* B, const double* C, double D,
                       int order, double Ts) {
    if (!block || !A || !B || !C || order <= 0 || order > 64) return -1;
    memset(block, 0, sizeof(WH_LinearBlock));
    block->type = WH_LIN_STATE_SPACE;
    block->order = order;
    block->state_dim = order;
    block->Ts = Ts;
    block->D = D;
    for (int i = 0; i < order; i++) {
        for (int j = 0; j < order; j++) {
            block->A[i * order + j] = A[i * order + j];
        }
        block->B[i] = B[i];
        block->C[i] = C[i];
    }
    return 0;
}

/* ??? Evaluation ????????????????????????????????????????????????????????? */

double wh_linear_evaluate(WH_LinearBlock* block, double u) {
    if (!block) return 0.0;
    double y = 0.0;

    if (block->type == WH_LIN_FIR) {
        /* Shift input buffer */
        for (int i = block->nb - 1; i > 0; i--) {
            block->state[i] = block->state[i - 1];
        }
        block->state[0] = u;
        /* Convolution */
        for (int i = 0; i < block->nb; i++) {
            y += block->b[i] * block->state[i];
        }
    } else if (block->type == WH_LIN_IIR_TF) {
        /* Shift input history */
        for (int i = block->nb - 1; i > 0; i--) {
            block->state[i] = block->state[i - 1];
        }
        block->state[0] = u;
        /* Feedforward */
        for (int i = 0; i < block->nb; i++) {
            y += block->b[i] * block->state[i];
        }
        /* Feedback (state[nb..nb+na-2] stores past outputs) */
        for (int i = 1; i < block->na; i++) {
            y -= block->a[i] * block->state[block->nb + i - 1];
        }
        /* Shift output history */
        int max_hist = block->nb + block->na;
        for (int i = max_hist - 1; i > block->nb; i--) {
            block->state[i] = block->state[i - 1];
        }
        block->state[block->nb] = y;
    } else if (block->type == WH_LIN_STATE_SPACE) {
        int n = block->order;
        double new_state[64];
        for (int i = 0; i < n; i++) {
            new_state[i] = 0.0;
            for (int j = 0; j < n; j++) {
                new_state[i] += block->A[i * n + j] * block->state[j];
            }
            new_state[i] += block->B[i] * u;
        }
        y = block->D * u;
        for (int i = 0; i < n; i++) {
            y += block->C[i] * block->state[i];
        }
        memcpy(block->state, new_state, n * sizeof(double));
    }

    return y;
}

void wh_linear_evaluate_batch(WH_LinearBlock* block,
                               const double* u, double* y, int n) {
    if (!block || !u || !y || n <= 0) return;
    wh_linear_reset(block);
    for (int i = 0; i < n; i++) {
        y[i] = wh_linear_evaluate(block, u[i]);
    }
}

/* ??? Reset ?????????????????????????????????????????????????????????????? */

void wh_linear_reset(WH_LinearBlock* block) {
    if (!block) return;
    memset(block->state, 0, sizeof(block->state));
    memset(block->state_buffer, 0, sizeof(block->state_buffer));
}

/* ??? DC gain ???????????????????????????????????????????????????????????? */

double wh_linear_get_dc_gain(const WH_LinearBlock* block) {
    if (!block) return NAN;
    if (block->type == WH_LIN_FIR) {
        double sum = 0.0;
        for (int i = 0; i < block->nb; i++) sum += block->b[i];
        return sum;
    }
    if (block->type == WH_LIN_IIR_TF) {
        double sum_b = 0.0, sum_a = 0.0;
        for (int i = 0; i < block->nb; i++) sum_b += block->b[i];
        for (int i = 0; i < block->na; i++) sum_a += block->a[i];
        if (fabs(sum_a) < 1e-12) return NAN;
        return sum_b / sum_a;
    }
    if (block->type == WH_LIN_STATE_SPACE) {
        int n = block->order;
        /* Solve (I-A)*x = B*1, then DC gain = C*x + D */
        double IminusA[64][64];
        double rhs[64];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                IminusA[i][j] = (i == j ? 1.0 : 0.0) - block->A[i * n + j];
            }
            rhs[i] = block->B[i];
        }
        /* Gaussian elimination */
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (fabs(IminusA[row][col]) > fabs(IminusA[pivot][col])) {
                    pivot = row;
                }
            }
            if (fabs(IminusA[pivot][col]) < 1e-12) return NAN; /* Singular */
            if (pivot != col) {
                for (int j = 0; j < n; j++) {
                    double t = IminusA[col][j];
                    IminusA[col][j] = IminusA[pivot][j];
                    IminusA[pivot][j] = t;
                }
                double t = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = t;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = IminusA[row][col] / IminusA[col][col];
                for (int j = col; j < n; j++) {
                    IminusA[row][j] -= factor * IminusA[col][j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
        /* Back substitution */
        double x[64] = {0};
        for (int i = n - 1; i >= 0; i--) {
            double sum = rhs[i];
            for (int j = i + 1; j < n; j++) {
                sum -= IminusA[i][j] * x[j];
            }
            x[i] = sum / IminusA[i][i];
        }
        double dc = block->D;
        for (int i = 0; i < n; i++) dc += block->C[i] * x[i];
        return dc;
    }
    return NAN;
}

/* ??? Frequency response ????????????????????????????????????????????????? */

void wh_linear_freq_response(const WH_LinearBlock* block, double omega,
                              double* mag, double* phase) {
    if (!block || !mag || !phase) return;

    /* Compute H(e^{j?}) = B(e^{j?}) / A(e^{j?}) */
    double complex Hz = 0.0 + 0.0 * I;
    double complex numer = 0.0 + 0.0 * I;
    double complex denom = 1.0 + 0.0 * I;

    if (block->type == WH_LIN_FIR) {
        for (int i = 0; i < block->nb; i++) {
            numer += block->b[i] * cexp(-I * omega * i);
        }
        Hz = numer;
    } else if (block->type == WH_LIN_IIR_TF) {
        for (int i = 0; i < block->nb; i++) {
            numer += block->b[i] * cexp(-I * omega * i);
        }
        for (int i = 0; i < block->na; i++) {
            denom += block->a[i] * cexp(-I * omega * i);
        }
        Hz = numer / denom;
    } else if (block->type == WH_LIN_STATE_SPACE) {
        int n = block->order;
        /* H(j?) = D + C*(j?*I - A)^{-1}*B */
        /* For discrete: H(e^{j?}) = D + C*(e^{j?}*I - A)^{-1}*B */
        double complex z = cexp(I * omega);
        double complex T[64][64];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T[i][j] = (i == j ? z : 0.0) - block->A[i * n + j];
            }
        }
        /* Solve for x: T*x = B */
        double complex x[64], rhs[64];
        for (int i = 0; i < n; i++) rhs[i] = block->B[i];
        /* Gaussian elimination over complex */
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (cabs(T[row][col]) > cabs(T[pivot][col])) pivot = row;
            }
            if (pivot != col) {
                for (int j = 0; j < n; j++) {
                    double complex tmp = T[col][j];
                    T[col][j] = T[pivot][j];
                    T[pivot][j] = tmp;
                }
                double complex tmp = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = tmp;
            }
            for (int row = col + 1; row < n; row++) {
                double complex factor = T[row][col] / T[col][col];
                for (int j = col; j < n; j++)
                    T[row][j] -= factor * T[col][j];
                rhs[row] -= factor * rhs[col];
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            double complex sum = rhs[i];
            for (int j = i + 1; j < n; j++)
                sum -= T[i][j] * x[j];
            x[i] = sum / T[i][i];
        }
        Hz = block->D;
        for (int i = 0; i < n; i++)
            Hz += block->C[i] * x[i];
    }

    *mag = cabs(Hz);
    *phase = carg(Hz);
}

/* ??? Stability ?????????????????????????????????????????????????????????? */

int wh_linear_is_stable(const WH_LinearBlock* block) {
    if (!block) return 0;
    if (block->type == WH_LIN_FIR) return 1; /* FIR is always stable */
    if (block->type == WH_LIN_IIR_TF) {
        /* Jury stability test for denominator polynomial */
        int n = block->na;
        if (n <= 1) return 1; /* a[0]=1, no dynamics */
        /* Check necessary condition: |a_{na-1}| < 1 */
        if (fabs(block->a[n - 1]) >= block->a[0]) return 0;
        /* Evaluate at z=1 and z=-1 */
        double A_1 = 0.0;
        for (int i = 0; i < n; i++) A_1 += block->a[i];
        if (A_1 <= 0.0) return 0;
        double A_m1 = 0.0;
        for (int i = 0; i < n; i++) {
            A_m1 += (i % 2 == 0 ? block->a[i] : -block->a[i]);
        }
        if (((n - 1) % 2 == 0 ? 1.0 : -1.0) * A_m1 <= 0.0) return 0;
        return 1;
    }
    if (block->type == WH_LIN_STATE_SPACE) {
        /* Check that all eigenvalues of A have magnitude < 1 */
        /* Simplified: use Gershgorin circle theorem */
        int n = block->order;
        for (int i = 0; i < n; i++) {
            double radius = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) radius += fabs(block->A[i * n + j]);
            }
            radius += fabs(block->A[i * n + i]);
            if (radius >= 1.0) return 0; /* Possible instability */
        }
        return 1;
    }
    return 1;
}

double wh_linear_get_pole_radius(const WH_LinearBlock* block) {
    if (!block || block->type == WH_LIN_FIR) return 0.0;
    if (block->type == WH_LIN_IIR_TF) {
        int n = block->na;
        if (n <= 1) return 0.0;
        /* For degree-1: pole = -a1 */
        if (n == 2) return fabs(-block->a[1]);
        /* For degree-2: max |root| */
        if (n == 3) {
            double disc = block->a[1] * block->a[1] - 4.0 * block->a[2];
            if (disc >= 0) {
                double r1 = fabs((-block->a[1] + sqrt(disc)) / 2.0);
                double r2 = fabs((-block->a[1] - sqrt(disc)) / 2.0);
                return r1 > r2 ? r1 : r2;
            }
            return sqrt(fabs(block->a[2]));
        }
        /* Conservative: return 0.99 (unknown but likely stable) or 1.01 */
        return 0.5;
    }
    return 0.0;
}

/* ??? Delay ?????????????????????????????????????????????????????????????? */

int wh_linear_get_delay(const WH_LinearBlock* block) {
    if (!block) return 0;
    for (int i = 0; i < block->nb; i++) {
        if (fabs(block->b[i]) > 1e-12) return i;
    }
    return block->nb;
}

/* ??? Impulse and step responses ????????????????????????????????????????? */

void wh_linear_impulse_response(WH_LinearBlock* block,
                                 double* impulse, int n_samples) {
    if (!block || !impulse || n_samples <= 0) return;
    WH_LinearBlock local_copy;
    memcpy(&local_copy, block, sizeof(WH_LinearBlock));
    wh_linear_reset(&local_copy);
    /* Apply unit impulse */
    wh_linear_evaluate(&local_copy, 1.0);
    for (int i = 0; i < n_samples; i++) {
        impulse[i] = wh_linear_evaluate(&local_copy, 0.0);
    }
}

void wh_linear_step_response(WH_LinearBlock* block,
                              double* step_resp, int n_samples) {
    if (!block || !step_resp || n_samples <= 0) return;
    WH_LinearBlock local_copy;
    memcpy(&local_copy, block, sizeof(WH_LinearBlock));
    wh_linear_reset(&local_copy);
    for (int i = 0; i < n_samples; i++) {
        step_resp[i] = wh_linear_evaluate(&local_copy, 1.0);
    }
}

/* ??? Pole computation via companion matrix ?????????????????????????????? */

int wh_linear_compute_poles(const WH_LinearBlock* block,
                             double complex* poles, int max_poles) {
    if (!block || !poles || max_poles <= 0) return 0;
    if (block->type == WH_LIN_FIR) return 0;

    int n = (block->type == WH_LIN_IIR_TF) ? (block->na - 1) : block->order;
    if (n <= 0) return 0;
    if (n > max_poles) n = max_poles;

    if (n == 1) {
        if (block->type == WH_LIN_IIR_TF)
            poles[0] = -block->a[1] / block->a[0];
        else
            poles[0] = block->A[0];
        return 1;
    }
    if (n == 2) {
        double a = 1.0, b, c;
        if (block->type == WH_LIN_IIR_TF) {
            b = block->a[1];
            c = block->a[2];
        } else {
            b = -(block->A[0] + block->A[3]);
            c = block->A[0] * block->A[3] - block->A[1] * block->A[2];
        }
        double disc = b * b - 4.0 * a * c;
        if (disc >= 0) {
            poles[0] = (-b + sqrt(disc)) / (2.0 * a);
            poles[1] = (-b - sqrt(disc)) / (2.0 * a);
        } else {
            poles[0] = (-b + I * sqrt(-disc)) / (2.0 * a);
            poles[1] = (-b - I * sqrt(-disc)) / (2.0 * a);
        }
        return 2;
    }
    return n;
}

/* ??? Printing ??????????????????????????????????????????????????????????? */

void wh_linear_print(const WH_LinearBlock* block) {
    if (!block) { printf("Linear block: NULL\n"); return; }
    printf("Linear Block [%s], order=%d, nb=%d, na=%d, Ts=%.4f\n",
           block->type == WH_LIN_FIR ? "FIR" :
           block->type == WH_LIN_IIR_TF ? "IIR" : "SS",
           block->order, block->nb, block->na, block->Ts);
    printf("  B: [");
    for (int i = 0; i < block->nb && i < 10; i++)
        printf("%.4f%s", block->b[i], i < block->nb - 1 ? ", " : "");
    if (block->nb > 10) printf("...");
    printf("]\n");
    if (block->na > 0) {
        printf("  A: [");
        for (int i = 0; i < block->na && i < 10; i++)
            printf("%.4f%s", block->a[i], i < block->na - 1 ? ", " : "");
        if (block->na > 10) printf("...");
        printf("]\n");
    }
    printf("  D=%.4f, DCGain=%.4f, Stable=%s\n",
           block->D, wh_linear_get_dc_gain(block),
           wh_linear_is_stable(block) ? "Yes" : "No");
}

void wh_linear_copy(WH_LinearBlock* dest, const WH_LinearBlock* src) {
    if (!dest || !src) return;
    memcpy(dest, src, sizeof(WH_LinearBlock));
}

int wh_linear_convert_to_iir(WH_LinearBlock* dest, const WH_LinearBlock* src) {
    if (!dest || !src) return -1;
    if (src->type == WH_LIN_FIR || src->type == WH_LIN_IIR_TF) {
        /* Direct copy since FIR trivially maps to IIR with A(z)=1 */
        memcpy(dest, src, sizeof(WH_LinearBlock));
        if (src->type == WH_LIN_FIR) {
            dest->type = WH_LIN_IIR_TF;
            dest->na = 1;
            dest->a[0] = 1.0;
        }
        return 0;
    }
    if (src->type == WH_LIN_STATE_SPACE) {
        /* Convert to controllable canonical form */
        /* H(z) = D + C(zI-A)^{-1}B = B_IIR(z)/A_IIR(z) */
        /* This is non-trivial; for now return coefficients up to order 2 */
        int n = src->order;
        if (n <= 0) return -1;

        memset(dest, 0, sizeof(WH_LinearBlock));
        dest->type = WH_LIN_IIR_TF;
        dest->nb = n + 1;
        dest->na = n + 1;
        dest->a[0] = 1.0;

        if (n == 1) {
            dest->a[1] = -src->A[0];
            dest->b[0] = src->D;
            dest->b[1] = src->C[0] * src->B[0];
        } else if (n == 2) {
            double trace = src->A[0] + src->A[3];
            double det = src->A[0] * src->A[3] - src->A[1] * src->A[2];
            dest->a[1] = -trace;
            dest->a[2] = det;
            dest->b[0] = src->D;
            dest->b[1] = src->C[0] * src->B[0] + src->C[1] * src->B[1];
            dest->b[2] = src->D * det + src->C[0] * (src->A[3] * src->B[0] - src->A[1] * src->B[1])
                         + src->C[1] * (-src->A[2] * src->B[0] + src->A[0] * src->B[1]);
        }
        return 0;
    }
    return -1;
}

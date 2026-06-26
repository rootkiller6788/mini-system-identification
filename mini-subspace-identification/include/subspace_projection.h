#ifndef SUBSPACE_PROJECTION_H
#define SUBSPACE_PROJECTION_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- Orthogonal and Oblique Projections
 *
 * Implements the geometric projection operations that form the core of
 * subspace identification algorithms. Given matrices A (data), B (basis
 * to project onto), and C (basis to project along), the projections are:
 *
 * Orthogonal projection of the row space of A onto the row space of B:
 *   A / B = A B^T (B B^T)^+ B
 *   A / B^bot = A (I - B^T (B B^T)^+ B)
 *
 * Oblique projection of A along B onto C:
 *   A /_B C = [A / B^bot] [C / B^bot]^+ C
 *
 * Implementation uses QR factorization of [B; C] for numerical stability
 * rather than computing pseudoinverses explicitly.
 *
 * Key formulas (Van Overschee & De Moor, 1996, Sec. 1.4):
 *   Given the LQ decomposition of [B; C; A]:
 *     [B]   [L_{11}  0       0     ] [Q_1^T]
 *     [C] = [L_{21}  L_{22}  0     ] [Q_2^T]
 *     [A]   [L_{31}  L_{32}  L_{33}] [Q_3^T]
 *
 *   Orthogonal projection: A / B^bot = L_{32} Q_2^T + L_{33} Q_3^T
 *   Oblique projection:    A /_B C = L_{32} L_{22}^+ [L_{21} L_{11}^{-1}  I] [B; C]
 *
 * References:
 *   Van Overschee & De Moor (1996) -- Subspace Identification, Ch. 1
 *   Katayama (2005) -- Subspace Methods for System Identification, Ch. 2
 *   Golub & Van Loan (2013) -- Matrix Computations, Sec. 5.3
 * ============================================================================ */

/* --- Orthogonal Projection ---
 * Projects row space of A onto row space of B:
 * A/B = A * B^T * (B * B^T)^+ * B
 * Output P has same dimensions as A. */
int subspace_project_onto(const SubspaceMatrix *A, const SubspaceMatrix *B,
                           SubspaceMatrix *P);

/* --- Orthogonal Complement Projection ---
 * Projects row space of A onto orthogonal complement of row space of B:
 * A/B^bot = A * (I - B^T * (B * B^T)^+ * B)
 * Output P has same dimensions as A. */
int subspace_project_onto_complement(const SubspaceMatrix *A,
                                      const SubspaceMatrix *B,
                                      SubspaceMatrix *P);

/* --- Oblique Projection ---
 * Projects A along the row space of B onto the row space of C:
 * A /_B C
 * Output O has same dimensions as A. */
int subspace_project_oblique(const SubspaceMatrix *A, const SubspaceMatrix *B,
                              const SubspaceMatrix *C, SubspaceMatrix *O);

/* --- Oblique Projection via LQ (Numerically Stable) ---
 * Computes oblique projection using the LQ decomposition of [B; C; A].
 * This is the preferred implementation for subspace identification. */
int subspace_project_oblique_lq(const SubspaceMatrix *Uf,
                                 const SubspaceMatrix *Wp,
                                 const SubspaceMatrix *Yf,
                                 SubspaceMatrix *O_i);

/* --- Weighting Matrix Construction ---
 * Constructs the left and right weighting matrices W_1 and W_2
 * used in the weighted SVD: W_1 * O_i * W_2 = U * Sigma * V^T */

/* N4SID weighting: W_1 = I, W_2 = I */
int subspace_weight_n4sid(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                           SubspaceMatrix *W1, SubspaceMatrix *W2);

/* MOESP weighting: W_1 = I, W_2 = Pi_{U_f^bot} */
int subspace_weight_moesp(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                           SubspaceMatrix *W1, SubspaceMatrix *W2);

/* CVA weighting: W_1 = (Y_f Pi_{U_f^bot} Y_f^T)^{-1/2}, W_2 = Pi_{U_f^bot} */
int subspace_weight_cva(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                         SubspaceMatrix *W1, SubspaceMatrix *W2);

/* Apply weighting: O_weighted = W1 * O * W2 */
int subspace_apply_weighting(const SubspaceMatrix *O_i,
                              const SubspaceMatrix *W1,
                              const SubspaceMatrix *W2,
                              SubspaceMatrix *O_weighted);

/* --- Extended Observability Matrix Recovery ---
 * From SVD: W_1 * O_i * W_2 = U * Sigma * V^T
 * Recover Gamma_i = W_1^{-1} * U * Sigma^{1/2} (up to similarity transform) */
int subspace_recover_gamma(const SubspaceMatrix *U_svd,
                            const double *singular_values, int n,
                            const SubspaceMatrix *W1,
                            SubspaceMatrix *Gamma_i);

/* --- State Sequence Recovery ---
 * From SVD: X_i = Sigma^{1/2} * V^T * W_2^{-1} * W_p (for certain weightings)
 * or X_i = Gamma_i^+ * O_i
 * where X_i = [x(i), x(i+1), ..., x(i+j-1)] */
int subspace_recover_state(const SubspaceMatrix *Gamma_i,
                            const SubspaceMatrix *O_i,
                            SubspaceMatrix *X_i);

/* Recover X_{i+1} using Gamma_{i-1} (Gamma without last block row) */
int subspace_recover_state_next(const SubspaceMatrix *Gamma_i,
                                 const SubspaceMatrix *O_i_shifted,
                                 SubspaceMatrix *X_ip1);

#endif /* SUBSPACE_PROJECTION_H */

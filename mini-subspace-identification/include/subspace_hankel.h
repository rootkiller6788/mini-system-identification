#ifndef SUBSPACE_HANKEL_H
#define SUBSPACE_HANKEL_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- Block Hankel Matrix Construction
 *
 * Implements construction of block Hankel matrices from time series data,
 * conversion between Hankel and standard matrix representations, and
 * compact storage schemes for large-scale data.
 *
 * For a signal s(k) with dimension d (d = m outputs or r inputs) and
 * N samples, the block Hankel matrix with i block rows and j columns is:
 *
 *   H_{i,j} = [ s(0)    s(1)    ... s(j-1)    ]
 *             [ s(1)    s(2)    ... s(j)      ]
 *             [ ...     ...     ... ...       ]
 *             [ s(i-1)  s(i)    ... s(i+j-2)  ]
 *
 * Each s(k) is a column vector of length d.
 * Dimensions: H is (d*i) x j where j = N - i + 1.
 *
 * The data equation in subspace identification:
 *   Y_f = Gamma_i X_f + H_i^d U_f + G_i E_f
 *
 * where Y_f and U_f are block Hankel matrices of outputs and inputs,
 * and W_p = [U_p; Y_p] is the past data matrix.
 *
 * References:
 *   Van Overschee & De Moor (1996) -- Subspace Identification, Ch. 2
 *   Katayama (2005) -- Subspace Methods for System Identification, Ch. 6
 * ============================================================================ */

/* --- Block Hankel Construction --- */

/* Construct from single-channel signal */
void subspace_hankel_build(const double *signal, int N, int block_rows,
                            SubspaceHankel *H);

/* Construct from multi-channel signal (MIMO) */
void subspace_hankel_build_mimo(const double *signal, int N, int dim,
                                 int block_rows, SubspaceHankel *H);

/* Build U_p, U_f, Y_p, Y_f from IO data with given block row count i */
int subspace_hankel_build_all(const SubspaceData *data, int i,
                               SubspaceHankel *Up, SubspaceHankel *Uf,
                               SubspaceHankel *Yp, SubspaceHankel *Yf);

/* Build the past data matrix W_p = [U_p; Y_p] */
void subspace_hankel_build_wp(const SubspaceHankel *Up, const SubspaceHankel *Yp,
                               SubspaceMatrix *Wp);

/* Build the future data matrix W_f = [U_f; Y_f] */
void subspace_hankel_build_wf(const SubspaceHankel *Uf, const SubspaceHankel *Yf,
                               SubspaceMatrix *Wf);

/* --- Hankel <-> Matrix Conversion --- */

/* Convert a block Hankel to a standard dense matrix (row-major output) */
void subspace_hankel_export_matrix(const SubspaceHankel *H, SubspaceMatrix *M,
                                    bool row_major);

/* Import from matrix into Hankel structure (reverse indexing) */
void subspace_hankel_import_matrix(const SubspaceMatrix *M, int dim_per_block,
                                    SubspaceHankel *H);

/* --- Hankel Shift Operations --- */

/* Shift a block Hankel matrix: remove first block row, add one at the end.
 * Used for constructing X_{i+1} from X_i in N4SID. */
void subspace_hankel_shift(const SubspaceHankel *H, SubspaceHankel *H_shifted);

/* Extract a sub-block: rows [start_block, start_block+num_blocks) */
void subspace_hankel_subblock(const SubspaceHankel *H, int start_block,
                               int num_blocks, SubspaceHankel *sub);

/* --- Data Preprocessing --- */

/* Remove mean and/or trend from IO data */
void subspace_data_detrend(SubspaceData *data, bool remove_mean,
                            bool remove_linear);

/* Scale data to unit variance */
void subspace_data_normalize(SubspaceData *data);

/* Validate data for subspace identification */
int subspace_data_validate(const SubspaceData *data);

#endif /* SUBSPACE_HANKEL_H */

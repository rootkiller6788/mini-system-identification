/**
 * clid_joint_io.h — Joint Input-Output Closed-Loop Identification
 *
 * The joint input-output approach treats the closed-loop system as a
 * multivariable open-loop system from (r,e) to (u,y), then recovers
 * plant and noise models from the joint transfer matrix.
 *
 * Core idea: Under feedback u = r - C y (or similar), the joint system is:
 *   [ y ]   [ G S_o    H_0 S_o   ] [ r ]
 *   [ u ] = [ S_o      -C H_0 S_o ] [ e ]
 *   where S_o = 1/(1+CG) is the output sensitivity.
 *
 * By identifying the 2x2 (or (ny+nu)x(nr+1)) transfer matrix, we can
 * recover G, C, and H by solving polynomial identities.
 *
 * Key advantage: NO knowledge of controller required! The controller
 * can be identified simultaneously with the plant.
 *
 * References:
 *   Van den Hof et al. (1995) Automatica 31(12)
 *   Schrama (1992) IEEE TAC 37(7)
 *   Ljung (1999) Ch.13.6
 */
#ifndef CLID_JOINT_IO_H
#define CLID_JOINT_IO_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Joint input-output identification via spectral analysis.
 *
 * Estimates the joint transfer matrix from spectral densities:
 *   G_joint(e^{jw}) = Phi_{zy}(w) * Phi_z(w)^{-1}
 * where z = [r; e_approx] is the "joint input".
 *
 * Uses Blackman-Tukey spectral estimation with user-specified window.
 * Recovers G(q) from the identified joint system by solving:
 *   G_yr = G * S_o  =>  G = G_yr * G_ur^{-1}
 * where G_yr is the estimated TF from r to y and G_ur from r to u.
 *
 * Reference: Schrama (1992); Van den Hof et al. (1995) Algorithm 2
 *
 * @param data      Closed-loop I/O data with known reference
 * @param opts      Options (controls window length, model order)
 * @param est_out   Output: identified plant + noise model
 * @return          0 on success
 */
int clid_joint_io_spectral(const CLID_Dataset *data,
                            const CLID_Options *opts,
                            CLID_Estimate *est_out);

/**
 * Joint input-output via correlation analysis.
 *
 * Uses cross-correlation functions instead of spectral densities:
 *   R_{z y}(tau) = (1/N) SUM_t z(t) y(t+tau)
 *   R_{z z}(tau) = (1/N) SUM_t z(t) z(t+tau)^T
 *
 * Then estimates impulse responses via:
 *   g_hat = R_{zz}^{-1} * R_{zy}
 * which is a linear regression in correlation space (COR method).
 *
 * Reference: Ljung (1999) Section 13.6; Soderstrom & Stoica (1989)
 */
int clid_joint_io_correlation(const CLID_Dataset *data,
                               const CLID_Options *opts,
                               CLID_TransferFcn *plant_ir);

/**
 * Joint IO via coprime factorization.
 *
 * Identifies coprime factors (N,D) of the plant G = N*D^{-1}
 * directly from closed-loop data without knowing the controller.
 *
 * Uses the fact that under feedback:
 *   D(q) u(t) - N(q) y(t) = D(q) r(t) + (filtered noise)
 *
 * This is a linear-in-parameters problem! The factors (N,D) can be
 * estimated by least squares on (u,y,r) data, yielding a consistent
 * estimate under mild conditions.
 *
 * Reference: Van den Hof & de Callafon (1996); Chou & Verhaegen (1997)
 */
int clid_joint_io_coprime(const CLID_Dataset *data,
                           const CLID_Options *opts,
                           CLID_TransferFcn *N,
                           CLID_TransferFcn *D);

/**
 * Recover plant and controller from identified joint model.
 *
 * Given the estimated joint transfer matrix:
 *   [ y ] = [ G_yr  G_ye ] [ r ]
 *   [ u ]   [ G_ur  G_ue ] [ e_approx ]
 *
 * Recover:
 *   G = G_yr * G_ur^{-1}  (plant from r-to-y and r-to-u paths)
 *   C = G_ur^{-1} - G    (controller, if feedback is u = r - C y)
 *
 * Handles MIMO case with pseudoinverse for non-square systems.
 */
int clid_joint_io_recover(const CLID_TransferFcn *G_yr,
                           const CLID_TransferFcn *G_ur,
                           CLID_TransferFcn *plant,
                           CLID_TransferFcn *controller);

/**
 * Two-stage / projection method for joint IO identification.
 *
 * Stage 1: Project u(t) onto r(t) (instrumental variable) to obtain
 *          u_hat(t) = projection of u onto reference space.
 *          This removes the noise-correlated component of u.
 *
 * Stage 2: Identify G from y(t) = G(q) u_hat(t) + H_0(q) e(t)
 *          using open-loop PEM on (u_hat, y). Since u_hat is
 *          uncorrelated with e (by construction), consistent!
 *
 * This is equivalent to instrumental variable with r as instrument.
 *
 * Reference: Van den Hof & Schrama (1993); Algorithm 3
 */
int clid_joint_io_projection(const CLID_Dataset *data,
                              const CLID_Options *opts,
                              CLID_Estimate *est_out);

#ifdef __cplusplus
}
#endif

#endif /* CLID_JOINT_IO_H */

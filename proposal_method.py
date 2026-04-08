from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

# NOTE:
# This variant disables both shrinkage and ridge by default.
# It uses the raw sample covariance blocks directly and will raise an
# error when the implied covariance is not positive definite/invertible.



# ============================================================
# Data containers
# ============================================================

@dataclass
class SelfTheta:
    lambda_x: np.ndarray
    theta_delta: np.ndarray
    lambda_y: np.ndarray
    theta_epsilon: np.ndarray
    tau: float

    def copy(self) -> "SelfTheta":
        return SelfTheta(
            lambda_x=self.lambda_x.copy(),
            theta_delta=self.theta_delta.copy(),
            lambda_y=self.lambda_y.copy(),
            theta_epsilon=self.theta_epsilon.copy(),
            tau=float(self.tau),
        )


@dataclass
class ProposedResult:
    beta_hat: float
    srmr_hat: float
    theta_hat: Dict[str, Any]
    sigma_hat: np.ndarray
    sigma_tilde_xy: np.ndarray
    delta_hat_xy: float
    cross_scale: float
    epsilon_n: float
    xi_n: float
    self_nll_hat: float
    cross_ratio_hat: float
    feasible_count: int
    total_self_candidates: int
    beta_interval_at_best_theta: Optional[Tuple[float, float]]
    diagnostics: Dict[str, Any]


# ============================================================
# Basic utilities
# ============================================================

def _as_2d_float(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return arr


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _sample_cov(A: np.ndarray) -> np.ndarray:
    Ac = _center(A)
    n = Ac.shape[0]
    if n < 2:
        raise ValueError("At least two observations are required.")
    S = (Ac.T @ Ac) / (n - 1)
    return 0.5 * (S + S.T)


def _sample_cov_blocks(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xc = _center(X)
    Yc = _center(Y)
    n = X.shape[0]
    denom = max(n - 1, 1)
    Sxx = (Xc.T @ Xc) / denom
    Syy = (Yc.T @ Yc) / denom
    Sxy = (Xc.T @ Yc) / denom
    Syx = Sxy.T
    return 0.5 * (Sxx + Sxx.T), 0.5 * (Syy + Syy.T), Sxy, Syx


def _safe_logdet_and_inv(Sigma: np.ndarray, jitter: float = 0.0, max_tries: int = 8) -> Tuple[float, np.ndarray, float]:
    """
    Compute log-determinant and inverse without automatic ridge regularization.

    If jitter <= 0, this function performs a strict single-shot evaluation on the
    input matrix itself. That means raw-data / no-ridge runs will fail as soon as
    the model-implied covariance is not positive definite or not invertible.

    If jitter > 0, the legacy progressive diagonal loading path is still
    available, but the no-regularization variant in this file sets jitter=0 by
    default everywhere.
    """
    current = Sigma.copy()

    if jitter <= 0:
        sign, logdet = np.linalg.slogdet(current)
        if sign <= 0:
            raise np.linalg.LinAlgError(
                "Covariance matrix is not positive definite and ridge=0."
            )
        inv = np.linalg.inv(current)
        return logdet, inv, 0.0

    eye = np.eye(Sigma.shape[0])
    used = 0.0
    for k in range(max_tries):
        sign, logdet = np.linalg.slogdet(current)
        if sign > 0:
            try:
                inv = np.linalg.inv(current)
                return logdet, inv, used
            except np.linalg.LinAlgError:
                pass
        used = jitter * (10 ** k)
        current = Sigma + used * eye
    raise np.linalg.LinAlgError("Failed to stabilize covariance matrix.")


def _build_sigma_x(lambda_x: np.ndarray, theta_delta: np.ndarray) -> np.ndarray:
    return np.outer(lambda_x, lambda_x) + np.diag(theta_delta)


def _build_sigma_y(lambda_y: np.ndarray, theta_epsilon: np.ndarray, tau: float) -> np.ndarray:
    return tau * np.outer(lambda_y, lambda_y) + np.diag(theta_epsilon)


def build_sigma_full(theta: SelfTheta, beta: float) -> np.ndarray:
    Sigma_xx = _build_sigma_x(theta.lambda_x, theta.theta_delta)
    Sigma_yy = _build_sigma_y(theta.lambda_y, theta.theta_epsilon, theta.tau)
    Sigma_xy = beta * np.outer(theta.lambda_x, theta.lambda_y)
    top = np.concatenate([Sigma_xx, Sigma_xy], axis=1)
    bottom = np.concatenate([Sigma_xy.T, Sigma_yy], axis=1)
    Sigma = np.concatenate([top, bottom], axis=0)
    return 0.5 * (Sigma + Sigma.T)


# ============================================================
# Self part: one-factor MLE with lambda_1 fixed to 1
# ============================================================

def _pack_self_params(
    lambda_x: np.ndarray,
    theta_delta: np.ndarray,
    lambda_y: np.ndarray,
    theta_epsilon: np.ndarray,
    tau: float,
) -> np.ndarray:
    return np.concatenate([
        lambda_x[1:],
        np.log(np.maximum(theta_delta, 1e-10)),
        lambda_y[1:],
        np.log(np.maximum(theta_epsilon, 1e-10)),
        np.array([np.log(max(tau, 1e-10))]),
    ])



def _shrink_cov(S: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Legacy diagonal-common-variance shrinkage kept for backward compatibility."""
    p = S.shape[0]
    mu = np.trace(S) / p
    return (1 - alpha) * S + alpha * mu * np.eye(p)


def _build_model_based_target_block(
    S: np.ndarray,
    loading_value: float = 0.7,
    reliability: float = 0.8,
) -> np.ndarray:
    """
    Construct the model-based shrinkage target described by
    De Jonckere & Rosseel for a *single-factor* measurement block.

    The paper defines the target in correlation metric as
        T* = Lambda* Psi* Lambda*' + Theta*
    with standardized factor loadings fixed at 0.7 and indicator reliability
    fixed at 0.8. To satisfy both requirements simultaneously in a one-factor
    block, we set
        lambda*_j = loading_value,
        theta*_jj = 1 - reliability,
        psi* = reliability / loading_value**2.
    This yields diag(T*) = 1 and off-diagonal entries equal to `reliability`.

    The target is then rescaled back to covariance metric via
        T = D T* D,  D = diag(sqrt(diag(S))).
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError('S must be a square covariance matrix.')
    if not (0.0 < loading_value < 1.0):
        raise ValueError('loading_value must lie in (0, 1).')
    if not (0.0 < reliability < 1.0):
        raise ValueError('reliability must lie in (0, 1).')

    p = S.shape[0]
    lam = np.full(p, float(loading_value))
    psi = float(reliability / (loading_value ** 2))
    theta_diag = np.full(p, 1.0 - reliability)

    T_star = psi * np.outer(lam, lam) + np.diag(theta_diag)
    T_star = 0.5 * (T_star + T_star.T)

    D = np.diag(np.sqrt(np.clip(np.diag(S), 1e-12, None)))
    T = D @ T_star @ D
    return 0.5 * (T + T.T)


def _apply_model_based_shrinkage(
    S: np.ndarray,
    lam: float,
    loading_value: float = 0.7,
    reliability: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return adjusted covariance and the corresponding model-based target."""
    if not (0.0 <= lam <= 1.0):
        raise ValueError('lam must lie in [0, 1].')
    T = _build_model_based_target_block(
        S,
        loading_value=loading_value,
        reliability=reliability,
    )
    S_adj = (1.0 - lam) * np.asarray(S, dtype=float) + lam * T
    return 0.5 * (S_adj + S_adj.T), T


def _unpack_self_params(z: np.ndarray, p1: int, p2: int) -> SelfTheta:
    idx = 0
    lambda_x = np.empty(p1)
    lambda_x[0] = 1.0
    lambda_x[1:] = z[idx:idx + p1 - 1]
    idx += p1 - 1

    theta_delta = np.exp(z[idx:idx + p1])
    idx += p1

    lambda_y = np.empty(p2)
    lambda_y[0] = 1.0
    lambda_y[1:] = z[idx:idx + p2 - 1]
    idx += p2 - 1

    theta_epsilon = np.exp(z[idx:idx + p2])
    idx += p2

    tau = float(np.exp(z[idx]))
    return SelfTheta(lambda_x, theta_delta, lambda_y, theta_epsilon, tau)


def self_negative_loglik(theta: SelfTheta, Sxx: np.ndarray, Syy: np.ndarray, ridge: float = 0.0) -> float:
    Sigma_xx = _build_sigma_x(theta.lambda_x, theta.theta_delta)
    Sigma_yy = _build_sigma_y(theta.lambda_y, theta.theta_epsilon, theta.tau)
    logdet_x, inv_x, _ = _safe_logdet_and_inv(Sigma_xx, jitter=ridge)
    logdet_y, inv_y, _ = _safe_logdet_and_inv(Sigma_yy, jitter=ridge)
    return float(logdet_x + np.trace(Sxx @ inv_x) + logdet_y + np.trace(Syy @ inv_y))




def normalize_sign(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    if v.ndim != 1:
        raise ValueError("v must be 1D.")
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v


def leading_eigenvector(S: np.ndarray) -> Tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eigh(np.asarray(S, dtype=float))
    vec = normalize_sign(vecs[:, -1])
    val = float(vals[-1])
    return vec, val


def make_cov_alpha_u(alpha: float, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    Sigma = float(alpha) * np.outer(u, u) + np.diag(np.asarray(theta, dtype=float))
    return 0.5 * (Sigma + Sigma.T)


def negloglik_factor_model_reference(params: np.ndarray, S: np.ndarray, p: int) -> float:
    w = np.asarray(params[:p], dtype=float)
    log_alpha = float(params[p])
    log_theta = np.asarray(params[p + 1:], dtype=float)
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-12:
        return 1e10
    u = normalize_sign(w / w_norm)
    alpha = float(np.exp(log_alpha))
    theta = np.exp(log_theta)
    Sigma = make_cov_alpha_u(alpha, u, theta)
    try:
        logdet, inv, _ = _safe_logdet_and_inv(Sigma, jitter=0.0)
    except np.linalg.LinAlgError:
        return 1e10
    val = logdet + np.trace(S @ inv)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def fit_one_factor_block_reference(
    S: np.ndarray,
    n_starts: int = 20,
    seed: Optional[int] = None,
    random_state: Optional[int] = None,
    enforce_tau_upper: bool = True,
    maxiter: int = 500,
) -> Dict[str, Any]:
    if random_state is not None and seed is not None:
        raise ValueError("Specify only one of seed or random_state.")
    if random_state is not None:
        seed = random_state

    rng = np.random.default_rng(seed)
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    u0, eval0 = leading_eigenvector(S)
    theta0 = np.maximum(np.diag(S) * 0.5, 1e-4)
    alpha0 = max(eval0 - float(np.mean(theta0)), 1e-4)

    def objective(params: np.ndarray) -> float:
        base = negloglik_factor_model_reference(params, S, p)
        if not enforce_tau_upper:
            return base
        w = np.asarray(params[:p], dtype=float)
        w_norm = float(np.linalg.norm(w))
        if w_norm < 1e-12:
            return 1e10
        u = normalize_sign(w / w_norm)
        alpha = float(np.exp(params[p]))
        upper = float(u.T @ S @ u)
        penalty = 0.0 if alpha <= upper + 1e-10 else 1e6 + 1e6 * (alpha - upper) ** 2
        return base + penalty

    starts = [np.concatenate([u0, [np.log(alpha0)], np.log(theta0)])]
    for _ in range(max(0, n_starts - 1)):
        w = u0 + 0.3 * rng.normal(size=p)
        a = np.log(max(alpha0 * np.exp(0.5 * rng.normal()), 1e-4))
        t = np.log(np.maximum(theta0 * np.exp(0.3 * rng.normal(size=p)), 1e-4))
        starts.append(np.concatenate([w, [a], t]))

    best = None
    best_fun = np.inf
    for x0 in starts:
        res = minimize(objective, x0=x0, method='L-BFGS-B', options={'maxiter': maxiter})
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best = res
    if best is None:
        raise RuntimeError('Reference tau estimation failed for all starts.')

    w_hat = np.asarray(best.x[:p], dtype=float)
    u_hat = normalize_sign(w_hat / max(np.linalg.norm(w_hat), 1e-12))
    alpha_hat = float(np.exp(best.x[p]))
    theta_hat = np.exp(np.asarray(best.x[p + 1:], dtype=float))
    Sigma_hat = make_cov_alpha_u(alpha_hat, u_hat, theta_hat)
    return {
        'u_hat': u_hat,
        'alpha_hat': alpha_hat,
        'theta_hat': theta_hat,
        'Sigma_hat': Sigma_hat,
        'objective': float(best_fun),
        'success': bool(best.success),
        'message': str(best.message),
        'proj_var': float(u_hat.T @ S @ u_hat),
    }


def recover_tau_reference(alpha_hat: float, u_hat: np.ndarray, Syy: np.ndarray) -> float:
    proj_var = float(np.asarray(u_hat, dtype=float).T @ np.asarray(Syy, dtype=float) @ np.asarray(u_hat, dtype=float))
    noise_level = float(np.mean(np.diag(np.asarray(Syy, dtype=float))))
    lambda_norm_sq_est = max(proj_var - noise_level, 1e-8)
    return float(alpha_hat / lambda_norm_sq_est)


def _lambda_from_direction_and_first_fixed(u: np.ndarray, first_loading_target: float = 1.0, min_first_abs: float = 0.25, max_abs_loading: float = 5.0) -> np.ndarray:
    u = normalize_sign(np.asarray(u, dtype=float))
    if abs(u[0]) < min_first_abs:
        u = u.copy()
        u[0] = min_first_abs
        u = normalize_sign(u)
    lam = u / max(u[0], 1e-12)
    lam[0] = first_loading_target
    lam[1:] = np.clip(lam[1:], -max_abs_loading, max_abs_loading)
    return lam


def _stable_loading_from_eigvec(
    S: np.ndarray,
    eigvec: np.ndarray,
    *,
    first_loading_target: float = 1.0,
    min_first_abs: float = 0.25,
    max_abs_loading: float = 2.5,
) -> np.ndarray:
    """
    Build a numerically stable one-factor loading initializer from the leading
    eigenvector without directly dividing by a potentially tiny first entry.

    Flow:
    1) align sign so the first element is non-negative,
    2) scale by infinity norm to avoid component explosion,
    3) if the first entry is too small, lift it to a minimum absolute level,
    4) rescale so the first loading becomes `first_loading_target`,
    5) clip the remaining entries to a conservative range.
    """
    v = np.asarray(eigvec, dtype=float).copy()
    if v.ndim != 1:
        raise ValueError('eigvec must be a 1D array.')

    if v[0] < 0:
        v = -v

    inf_norm = float(np.max(np.abs(v)))
    if not np.isfinite(inf_norm) or inf_norm < 1e-12:
        v = np.ones_like(v)
        inf_norm = 1.0
    v = v / inf_norm

    if abs(v[0]) < min_first_abs:
        v[0] = min_first_abs

    lam = v / v[0]
    lam[0] = first_loading_target
    lam[1:] = np.clip(lam[1:], -max_abs_loading, max_abs_loading)
    return lam



def _default_theta_from_loading(
    S: np.ndarray,
    lam: np.ndarray,
    scale: float,
    min_unique_ratio: float = 0.10,
    min_unique_abs: float = 1e-2,
) -> np.ndarray:
    diag_s = np.clip(np.diag(S), min_unique_abs, None)
    lower = np.maximum(min_unique_abs, min_unique_ratio * diag_s)
    raw = diag_s - scale * (lam ** 2)
    return np.maximum(raw, lower)



def _default_self_init(Sxx: np.ndarray, Syy: np.ndarray) -> SelfTheta:
    vals_x, vecs_x = np.linalg.eigh(Sxx)
    vals_y, vecs_y = np.linalg.eigh(Syy)
    vx = vecs_x[:, -1]
    vy = vecs_y[:, -1]

    lambda_x = _stable_loading_from_eigvec(Sxx, vx)
    lambda_y = _stable_loading_from_eigvec(Syy, vy)

    tau0 = float(np.clip(vals_y[-1], 1e-2, 10.0))
    theta_delta = _default_theta_from_loading(Sxx, lambda_x, scale=1.0)
    theta_epsilon = _default_theta_from_loading(Syy, lambda_y, scale=tau0)

    return SelfTheta(lambda_x, theta_delta, lambda_y, theta_epsilon, tau0)


def _theta_upper_from_cov(S: np.ndarray, theta_upper_scale: float, theta_lower: float) -> float:
    return float(max(theta_lower * 10.0, theta_upper_scale * np.max(np.diag(S))))


def _block_nll(
    S: np.ndarray,
    lam: np.ndarray,
    theta: np.ndarray,
    scale: float,
    ridge: float = 0.0,
) -> float:
    Sigma = scale * np.outer(lam, lam) + np.diag(theta)
    logdet, inv, _ = _safe_logdet_and_inv(Sigma, jitter=ridge)
    return float(logdet + np.trace(S @ inv))


def _update_theta_closed_form(
    S: np.ndarray,
    lam: np.ndarray,
    scale: float,
    theta_lower: float,
    theta_upper: float,
    theta_prev: Optional[np.ndarray] = None,
    relax: float = 0.35,
    min_unique_ratio: float = 0.10,
) -> np.ndarray:
    """
    Damped update for unique variances.
    Instead of replacing theta by the raw closed-form value in one shot,
    blend the new candidate with the previous iterate.
    """
    diag_s = np.clip(np.diag(S), theta_lower, None)
    lower_vec = np.maximum(theta_lower, min_unique_ratio * diag_s)
    raw = diag_s - scale * (lam ** 2)
    raw = np.minimum(np.maximum(raw, lower_vec), theta_upper)
    if theta_prev is None:
        return raw
    relax = float(np.clip(relax, 1e-3, 1.0))
    theta_prev = np.asarray(theta_prev, dtype=float)
    blended = (1.0 - relax) * theta_prev + relax * raw
    return np.minimum(np.maximum(blended, lower_vec), theta_upper)


def _alternating_block_optimize(
    S: np.ndarray,
    lam_init: np.ndarray,
    theta_init: np.ndarray,
    scale_init: float,
    *,
    fix_scale: bool,
    ridge: float,
    lambda_bound: float,
    theta_lower: float,
    theta_upper: float,
    tau_lower: float,
    tau_upper: float,
    alt_maxiter: int,
    alt_tol: float,
    polish_maxiter: int,
    theta_relax: float,
    theta_min_unique_ratio: float,
    fixed_scale_value: Optional[float] = None,
) -> Dict[str, Any]:
    p = S.shape[0]
    lam = np.asarray(lam_init, dtype=float).copy()
    theta = np.clip(np.asarray(theta_init, dtype=float).copy(), theta_lower, theta_upper)
    if fix_scale:
        if fixed_scale_value is None:
            scale = 1.0
        else:
            scale = float(np.clip(fixed_scale_value, tau_lower, tau_upper))
    else:
        scale = float(np.clip(scale_init, tau_lower, tau_upper))
    lam[0] = 1.0
    lam[1:] = np.clip(lam[1:], -lambda_bound, lambda_bound)

    history: List[float] = []
    alt_success = False
    last_message = 'alternating_maxiter_reached'

    for _ in range(max(1, alt_maxiter)):
        prev_obj = _block_nll(S, lam, theta, scale, ridge=ridge)
        theta = _update_theta_closed_form(
            S, lam, scale, theta_lower, theta_upper,
            theta_prev=theta, relax=theta_relax,
            min_unique_ratio=theta_min_unique_ratio,
        )

        if fix_scale:
            z0 = lam[1:].copy()
            bounds = [(-lambda_bound, lambda_bound)] * (p - 1)

            def obj_lambda(z: np.ndarray) -> float:
                lam_trial = np.empty(p)
                lam_trial[0] = 1.0
                lam_trial[1:] = z
                return _block_nll(S, lam_trial, theta, scale, ridge=ridge)

            res = minimize(obj_lambda, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': polish_maxiter})
            if getattr(res, 'x', None) is not None:
                lam[1:] = np.clip(np.asarray(res.x, dtype=float), -lambda_bound, lambda_bound)
            inner_success = bool(res.success and np.isfinite(res.fun))
            inner_message = str(res.message)
        else:
            z0 = np.concatenate([lam[1:], np.array([np.log(max(scale, tau_lower))])])
            bounds = [(-lambda_bound, lambda_bound)] * (p - 1) + [(np.log(tau_lower), np.log(tau_upper))]

            def obj_lambda_scale(z: np.ndarray) -> float:
                lam_trial = np.empty(p)
                lam_trial[0] = 1.0
                lam_trial[1:] = z[:-1]
                scale_trial = float(np.exp(z[-1]))
                return _block_nll(S, lam_trial, theta, scale_trial, ridge=ridge)

            res = minimize(obj_lambda_scale, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': polish_maxiter})
            if getattr(res, 'x', None) is not None:
                x = np.asarray(res.x, dtype=float)
                lam[1:] = np.clip(x[:-1], -lambda_bound, lambda_bound)
                scale = float(np.clip(np.exp(x[-1]), tau_lower, tau_upper))
            inner_success = bool(res.success and np.isfinite(res.fun))
            inner_message = str(res.message)

        theta = _update_theta_closed_form(
            S, lam, scale, theta_lower, theta_upper,
            theta_prev=theta, relax=theta_relax,
            min_unique_ratio=theta_min_unique_ratio,
        )
        cur_obj = _block_nll(S, lam, theta, scale, ridge=ridge)
        history.append(cur_obj)
        rel_improve = abs(prev_obj - cur_obj) / max(1.0, abs(prev_obj))
        if rel_improve <= alt_tol:
            alt_success = True
            last_message = f'converged_by_rel_improve<{alt_tol:g}'
            break
        last_message = inner_message if inner_success else f'inner_failure: {inner_message}'

    if fix_scale:
        z0 = np.concatenate([lam[1:], np.log(np.clip(theta, theta_lower, theta_upper))])
        bounds = [(-lambda_bound, lambda_bound)] * (p - 1) + [(np.log(theta_lower), np.log(theta_upper))] * p

        def obj_joint(z: np.ndarray) -> float:
            lam_trial = np.empty(p)
            lam_trial[0] = 1.0
            lam_trial[1:] = z[:p - 1]
            theta_trial = np.exp(z[p - 1:])
            return _block_nll(S, lam_trial, theta_trial, scale, ridge=ridge)

        res_polish = minimize(obj_joint, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': polish_maxiter})
        if getattr(res_polish, 'x', None) is not None:
            x = np.asarray(res_polish.x, dtype=float)
            lam[1:] = np.clip(x[:p - 1], -lambda_bound, lambda_bound)
            theta = np.clip(np.exp(x[p - 1:]), theta_lower, theta_upper)
    else:
        z0 = np.concatenate([
            lam[1:], np.log(np.clip(theta, theta_lower, theta_upper)), np.array([np.log(max(scale, tau_lower))]),
        ])
        bounds = [(-lambda_bound, lambda_bound)] * (p - 1) + [(np.log(theta_lower), np.log(theta_upper))] * p + [(np.log(tau_lower), np.log(tau_upper))]

        def obj_joint(z: np.ndarray) -> float:
            lam_trial = np.empty(p)
            lam_trial[0] = 1.0
            lam_trial[1:] = z[:p - 1]
            theta_trial = np.exp(z[p - 1:p - 1 + p])
            scale_trial = float(np.exp(z[-1]))
            return _block_nll(S, lam_trial, theta_trial, scale_trial, ridge=ridge)

        res_polish = minimize(obj_joint, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': polish_maxiter})
        if getattr(res_polish, 'x', None) is not None:
            x = np.asarray(res_polish.x, dtype=float)
            lam[1:] = np.clip(x[:p - 1], -lambda_bound, lambda_bound)
            theta = np.clip(np.exp(x[p - 1:p - 1 + p]), theta_lower, theta_upper)
            scale = float(np.clip(np.exp(x[-1]), tau_lower, tau_upper))

    final_obj = _block_nll(S, lam, theta, scale, ridge=ridge)
    return {
        'lambda': lam,
        'theta': theta,
        'scale': float(scale),
        'objective': float(final_obj),
        'alternating_success': bool(alt_success),
        'alternating_message': last_message,
        'alternating_iterations': int(len(history)),
        'alternating_history': history,
        'polish_success': bool(res_polish.success and np.isfinite(res_polish.fun)),
        'polish_message': str(res_polish.message),
        'polish_fun': float(res_polish.fun) if np.isfinite(res_polish.fun) else float(final_obj),
    }


def _optimize_single_start(task: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        start,
        Sxx,
        Syy,
        ridge,
        lambda_bound,
        theta_lower,
        theta_upper_scale,
        tau_lower,
        tau_upper,
        alt_maxiter,
        alt_tol,
        polish_maxiter,
        theta_relax,
        theta_min_unique_ratio,
        fixed_tau,
    ) = task

    try:
        theta_upper_x = _theta_upper_from_cov(Sxx, theta_upper_scale, theta_lower)
        theta_upper_y = _theta_upper_from_cov(Syy, theta_upper_scale, theta_lower)

        x_res = _alternating_block_optimize(
            Sxx, start.lambda_x, start.theta_delta, 1.0,
            fix_scale=True, fixed_scale_value=1.0,
            ridge=ridge, lambda_bound=lambda_bound,
            theta_lower=theta_lower, theta_upper=theta_upper_x,
            tau_lower=tau_lower, tau_upper=tau_upper,
            alt_maxiter=alt_maxiter, alt_tol=alt_tol,
            polish_maxiter=polish_maxiter,
            theta_relax=theta_relax,
            theta_min_unique_ratio=theta_min_unique_ratio,
        )
        if fixed_tau is None:
            y_res = _alternating_block_optimize(
                Syy, start.lambda_y, start.theta_epsilon, start.tau,
                fix_scale=False,
                ridge=ridge, lambda_bound=lambda_bound,
                theta_lower=theta_lower, theta_upper=theta_upper_y,
                tau_lower=tau_lower, tau_upper=tau_upper,
                alt_maxiter=alt_maxiter, alt_tol=alt_tol,
                polish_maxiter=polish_maxiter,
                theta_relax=theta_relax,
                theta_min_unique_ratio=theta_min_unique_ratio,
            )
            tau_hat = float(y_res['scale'])
        else:
            y_res = _alternating_block_optimize(
                Syy, start.lambda_y, start.theta_epsilon, fixed_tau,
                fix_scale=True, fixed_scale_value=float(fixed_tau),
                ridge=ridge, lambda_bound=lambda_bound,
                theta_lower=theta_lower, theta_upper=theta_upper_y,
                tau_lower=tau_lower, tau_upper=max(tau_upper, float(fixed_tau)),
                alt_maxiter=alt_maxiter, alt_tol=alt_tol,
                polish_maxiter=polish_maxiter,
                theta_relax=theta_relax,
                theta_min_unique_ratio=theta_min_unique_ratio,
            )
            tau_hat = float(fixed_tau)

        theta_hat = SelfTheta(
            lambda_x=np.asarray(x_res['lambda'], dtype=float),
            theta_delta=np.asarray(x_res['theta'], dtype=float),
            lambda_y=np.asarray(y_res['lambda'], dtype=float),
            theta_epsilon=np.asarray(y_res['theta'], dtype=float),
            tau=tau_hat,
        )
        final_obj = float(x_res['objective'] + y_res['objective'])
        success = bool(x_res['polish_success'] and y_res['polish_success'])
        message = f"X[{x_res['polish_message']}] | Y[{y_res['polish_message']}]"
        return {
            'success': success,
            'fun': final_obj,
            'theta': theta_hat,
            'message': message,
            'x_info': x_res,
            'y_info': y_res,
        }
    except Exception as e:
        return {
            'success': False,
            'fun': np.inf,
            'theta': None,
            'message': f'{type(e).__name__}: {e}',
            'x_info': None,
            'y_info': None,
        }


def _fit_self_mle_core(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    n_starts: int = 10,
    random_state: Optional[int] = None,
    ridge: float = 0.0,
    maxiter: int = 2000,
    n_jobs: Optional[int] = None,
    parallel_backend: str = 'process',
    lambda_bound: float = 5.0,
    theta_lower: float = 1e-4,
    theta_upper_scale: float = 10.0,
    tau_lower: float = 1e-4,
    tau_upper: float = 100.0,
    alt_maxiter: int = 50,
    alt_tol: float = 1e-7,
    theta_relax: float = 0.35,
    theta_min_unique_ratio: float = 0.05,#0.10
    tau_strategy: str = 'joint',
    tau_reference_enforce_upper: bool = True,
) -> Tuple[SelfTheta, float, Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    p1, p2 = Sxx.shape[0], Syy.shape[0]
    base = _default_self_init(Sxx, Syy)
    tau_reference_info = None
    fixed_tau = None

    if tau_strategy not in {'joint', 'reference_then_lambda'}:
        raise ValueError("tau_strategy must be 'joint' or 'reference_then_lambda'.")

    if tau_strategy == 'reference_then_lambda':
        ref_fit = fit_one_factor_block_reference(
            Syy,
            n_starts=max(10, n_starts),
            seed=int(rng.integers(1_000_000)),
            enforce_tau_upper=tau_reference_enforce_upper,
        )
        tau_ref = recover_tau_reference(ref_fit['alpha_hat'], ref_fit['u_hat'], Syy)
        tau_ref = float(np.clip(tau_ref, tau_lower, tau_upper))
        base.lambda_y = _lambda_from_direction_and_first_fixed(ref_fit['u_hat'], max_abs_loading=lambda_bound)
        base.theta_epsilon = np.maximum(ref_fit['theta_hat'], theta_lower)
        base.tau = tau_ref
        fixed_tau = tau_ref
        tau_reference_info = {
            'fit': ref_fit,
            'tau_hat': float(tau_ref),
            'strategy': 'reference_one_factor_then_fixed_tau_lambda_mle',
        }

    starts = [base]
    tau_upper_eff = max(tau_upper, base.tau * 2.0, 1.0)

    for _ in range(max(0, n_starts - 1)):
        bx = base.lambda_x.copy()
        by = base.lambda_y.copy()
        bx[1:] += rng.normal(scale=0.2, size=p1 - 1)
        by[1:] += rng.normal(scale=0.2, size=p2 - 1)
        bx[1:] = np.clip(bx[1:], -lambda_bound, lambda_bound)
        by[1:] = np.clip(by[1:], -lambda_bound, lambda_bound)
        td = np.maximum(base.theta_delta * np.exp(rng.normal(scale=0.2, size=p1)), theta_lower)
        te = np.maximum(base.theta_epsilon * np.exp(rng.normal(scale=0.2, size=p2)), theta_lower)
        tau = float(base.tau if fixed_tau is not None else np.clip(base.tau * np.exp(rng.normal(scale=0.25)), tau_lower, tau_upper_eff))
        starts.append(SelfTheta(bx, td, by, te, tau))

    requested_jobs = (os.cpu_count() or 1) if n_jobs is None else int(n_jobs)
    requested_jobs = max(1, requested_jobs)
    n_workers = min(requested_jobs, len(starts))
    tasks = [
        (
            start, Sxx, Syy, ridge, lambda_bound, theta_lower, theta_upper_scale,
            tau_lower, tau_upper_eff, alt_maxiter, alt_tol, maxiter,
            theta_relax, theta_min_unique_ratio, fixed_tau,
        )
        for start in starts
    ]

    results: List[Dict[str, Any]] = []
    backend_used = 'serial'
    if n_workers == 1:
        results = [_optimize_single_start(task) for task in tasks]
    else:
        if parallel_backend != 'process':
            raise ValueError("parallel_backend must be 'process'.")
        backend_used = f'process:{n_workers}'
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_optimize_single_start, task) for task in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

    best_theta = None
    best_value = np.inf
    best_res = None
    success_count = 0
    fail_messages = []
    for res in results:
        if res['success'] and res['theta'] is not None and np.isfinite(res['fun']):
            success_count += 1
            if res['fun'] < best_value:
                best_value = float(res['fun'])
                best_theta = res['theta']
                best_res = res
        else:
            fail_messages.append(res['message'])

    if best_theta is None:
        joined = ' | '.join(fail_messages[:5]) if fail_messages else 'No successful starts.'
        raise RuntimeError(f'Self-part optimization failed for all starts. Details: {joined}')

    info = {
        'optimizer_success': True,
        'optimizer_message': str(best_res['message']),
        'n_starts': int(n_starts),
        'n_jobs': int(n_workers),
        'parallel_backend': backend_used,
        'successful_starts': int(success_count),
        'failed_starts': int(len(results) - success_count),
        'failed_messages': fail_messages,
        'objective_value': float(best_value),
        'method': 'reference_tau_then_lambda' if fixed_tau is not None else 'joint_tau_lambda',
        'tau_strategy': tau_strategy,
        'tau_reference': tau_reference_info,
        'bounds': {
            'lambda_bound': float(lambda_bound),
            'theta_lower': float(theta_lower),
            'theta_upper_scale': float(theta_upper_scale),
            'tau_lower': float(tau_lower),
            'tau_upper': float(tau_upper_eff),
        },
        'alternating': {
            'alt_maxiter': int(alt_maxiter),
            'alt_tol': float(alt_tol),
            'theta_relax': float(theta_relax),
            'theta_min_unique_ratio': float(theta_min_unique_ratio),
        },
        'best_start_details': {'x': best_res['x_info'], 'y': best_res['y_info']},
    }
    return best_theta, best_value, info


def fit_self_mle(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    n_starts: int = 10,
    random_state: Optional[int] = None,
    ridge: float = 0.0,
    maxiter: int = 2000,
    n_jobs: Optional[int] = None,
    parallel_backend: str = 'process',
    shrinkage_method: Optional[str] = None,
    shrinkage_lambda: Optional[float] = None,
    shrinkage_grid: Optional[List[float]] = None,
    shrinkage_apply_when: str = 'if_needed',
    shrinkage_loading_value: float = 0.7,
    shrinkage_reliability: float = 0.8,
    lambda_bound: float = 5.0,
    theta_lower: float = 1e-4,
    theta_upper_scale: float = 10.0,
    tau_lower: float = 1e-4,
    tau_upper: float = 100.0,
    alt_maxiter: int = 50,
    alt_tol: float = 1e-7,
    theta_relax: float = 0.35,
    theta_min_unique_ratio: float = 0.10,
    tau_strategy: str = 'reference_then_lambda',
    tau_reference_enforce_upper: bool = True,
    verbose: bool = True,
) -> Tuple[SelfTheta, float, Dict[str, Any]]:
    base_kwargs = dict(
        n_starts=n_starts,
        random_state=random_state,
        ridge=ridge,
        maxiter=maxiter,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        lambda_bound=lambda_bound,
        theta_lower=theta_lower,
        theta_upper_scale=theta_upper_scale,
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        alt_maxiter=alt_maxiter,
        alt_tol=alt_tol,
        theta_relax=theta_relax,
        theta_min_unique_ratio=theta_min_unique_ratio,
        tau_strategy=tau_strategy,
        tau_reference_enforce_upper=tau_reference_enforce_upper,
    )

    def _print_status(success: bool, lam: float, msg: str) -> None:
        if verbose:
            status = 'SUCCESS' if success else 'FAIL'
            print(f'[fit_self_mle] {status}: lambda={lam:.3f} | {msg}')

    if shrinkage_method is None:
        theta, value, info = _fit_self_mle_core(Sxx, Syy, **base_kwargs)
        info = dict(info)
        info.update({'shrinkage_method': None, 'selected_lambda': 0.0, 'target_x': None, 'target_y': None, 'Sxx_input': Sxx, 'Syy_input': Syy})
        _print_status(True, 0.0, info.get('optimizer_message', 'no message'))
        return theta, value, info

    if shrinkage_method != 'model_based':
        raise ValueError("Only shrinkage_method=None or 'model_based' is supported.")
    if shrinkage_apply_when not in {'always', 'if_needed'}:
        raise ValueError("shrinkage_apply_when must be 'always' or 'if_needed'.")

    if shrinkage_lambda is not None:
        Sxx_adj, Txx = _apply_model_based_shrinkage(Sxx, lam=float(shrinkage_lambda), loading_value=shrinkage_loading_value, reliability=shrinkage_reliability)
        Syy_adj, Tyy = _apply_model_based_shrinkage(Syy, lam=float(shrinkage_lambda), loading_value=shrinkage_loading_value, reliability=shrinkage_reliability)
        theta, value, info = _fit_self_mle_core(Sxx_adj, Syy_adj, **base_kwargs)
        info = dict(info)
        info.update({'shrinkage_method': 'model_based', 'selected_lambda': float(shrinkage_lambda), 'target_x': Txx, 'target_y': Tyy, 'Sxx_input': Sxx_adj, 'Syy_input': Syy_adj, 'lambda_selection': 'user_supplied'})
        _print_status(True, float(shrinkage_lambda), info.get('optimizer_message', 'no message'))
        return theta, value, info

    if shrinkage_grid is None:
        shrinkage_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    shrinkage_grid = sorted({float(x) for x in shrinkage_grid if 0.0 <= float(x) <= 1.0})
    if not shrinkage_grid:
        raise ValueError('shrinkage_grid must contain at least one value in [0, 1].')

    last_error = None
    candidate_lambdas = shrinkage_grid if shrinkage_apply_when == 'always' else ([0.0] + [x for x in shrinkage_grid if x != 0.0])
    for lam in candidate_lambdas:
        try:
            if lam == 0.0:
                Sxx_adj, Txx = np.asarray(Sxx, dtype=float), None
                Syy_adj, Tyy = np.asarray(Syy, dtype=float), None
            else:
                Sxx_adj, Txx = _apply_model_based_shrinkage(Sxx, lam=lam, loading_value=shrinkage_loading_value, reliability=shrinkage_reliability)
                Syy_adj, Tyy = _apply_model_based_shrinkage(Syy, lam=lam, loading_value=shrinkage_loading_value, reliability=shrinkage_reliability)
            theta, value, info = _fit_self_mle_core(Sxx_adj, Syy_adj, **base_kwargs)
            info = dict(info)
            info.update({'shrinkage_method': 'model_based', 'selected_lambda': float(lam), 'target_x': Txx, 'target_y': Tyy, 'Sxx_input': Sxx_adj, 'Syy_input': Syy_adj, 'lambda_selection': 'smallest_convergent_grid'})
            _print_status(True, float(lam), info.get('optimizer_message', 'no message'))
            return theta, value, info
        except Exception as exc:
            last_error = exc
            _print_status(False, float(lam), str(exc))

    if last_error is None:
        raise RuntimeError('Self-part optimization failed and no shrinkage candidate was tried.')
    raise RuntimeError('Self-part optimization failed for all tested shrinkage intensities. ' f'Last error: {last_error}')


# ============================================================
# Bootstrap thresholds
# ============================================================

def _resample_rows(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)
    return X[idx], Y[idx]


def estimate_epsilon_n(
    X: np.ndarray,
    Y: np.ndarray,
    theta_hat: SelfTheta,
    alpha: float = 0.05,
    B: int = 200,
    n_starts: int = 5,
    random_state: Optional[int] = None,
    ridge: float = 0.0,
    self_n_jobs: Optional[int] = None,
    tau_strategy: str = 'reference_then_lambda',
    tau_reference_enforce_upper: bool = True,
) -> Tuple[float, np.ndarray, List[SelfTheta]]:
    rng = np.random.default_rng(random_state)
    deltas = []
    accepted_thetas: List[SelfTheta] = []

    for _ in range(B):
        Xb, Yb = _resample_rows(X, Y, rng)
        Sxx_b, Syy_b, _, _ = _sample_cov_blocks(Xb, Yb)

        theta_b, nll_b, _ = fit_self_mle(
            Sxx_b, Syy_b,
            n_starts=n_starts,
            random_state=int(rng.integers(1_000_000)),
            ridge=ridge,
            n_jobs=self_n_jobs,
            shrinkage_method='model_based',
            shrinkage_lambda=None,
            shrinkage_apply_when='if_needed',
            tau_strategy=tau_strategy,
            tau_reference_enforce_upper=tau_reference_enforce_upper,
            verbose=False,
        )
        nll_main_on_boot = self_negative_loglik(theta_hat, Sxx_b, Syy_b, ridge=ridge)
        delta_b = max(0.0, float(nll_main_on_boot - nll_b))
        deltas.append(delta_b)
        accepted_thetas.append(theta_b)

    deltas_arr = np.asarray(deltas, dtype=float)
    epsilon_n = float(np.quantile(deltas_arr, 1.0 - alpha)) if len(deltas_arr) else 0.0
    return epsilon_n, deltas_arr, accepted_thetas


# ============================================================
# Cross part: ECDM-based thresholding
# ============================================================

def ECDM(X1: np.ndarray, X2: np.ndarray) -> float:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    if X1.ndim == 1:
        n = X1.shape[0]
        X1 = X1.reshape(1, n)
    else:
        n = X1.shape[1]

    if X2.ndim == 1:
        X2 = X2.reshape(1, n)

    if X2.shape[1] != n:
        raise ValueError("X1 and X2 must have the same sample size in columns.")
    if n < 4:
        raise ValueError("ECDM requires at least 4 observations.")

    n1 = int(np.ceil(n / 2))
    n2 = n - n1
    if n1 <= 1 or n2 <= 1:
        raise ValueError("ECDM split sizes are too small; increase n.")

    u = 2 * n1 * n2 / ((n1 - 1) * (n2 - 1) * n * (n - 1))
    W = 0.0

    def V1(k: int, X: np.ndarray) -> np.ndarray:
        half = int(np.floor(k / 2))
        if half >= n1:
            index = np.arange(half - n1, half).astype(int)
        else:
            index = np.concatenate([np.arange(half), np.arange(half + n2, n)]).astype(int)
        return X[:, index]

    def V2(k: int, X: np.ndarray) -> np.ndarray:
        half = int(np.floor(k / 2))
        if half <= n1:
            index = np.arange(half, half + n2).astype(int)
        else:
            index = np.concatenate([np.arange(half - n1), np.arange(half, n)]).astype(int)
        return X[:, index]

    def H1_1(k: int) -> np.ndarray:
        return V1(k, X1).mean(axis=1)

    def H2_1(k: int) -> np.ndarray:
        return V2(k, X1).mean(axis=1)

    def H1_2(k: int) -> np.ndarray:
        return V1(k, X2).mean(axis=1)

    def H2_2(k: int) -> np.ndarray:
        return V2(k, X2).mean(axis=1)

    S = np.arange(3, 2 * n)
    M1 = [list(map(H1_1, S)), list(map(H1_2, S))]
    M2 = [list(map(H2_1, S)), list(map(H2_2, S))]

    def q_corr(i: int, j: int) -> float:
        q1 = np.dot(X1[:, i - 1] - M1[0][i + j - 3], X1[:, j - 1] - M2[0][i + j - 3])
        q2 = np.dot(X2[:, i - 1] - M1[1][i + j - 3], X2[:, j - 1] - M2[1][i + j - 3])
        return float(q1 * q2)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            W += q_corr(i, j)

    Tn = float(W * u)
    return max(Tn, 1e-12)


def sparse_cross_cov(X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    p1, n = X1.shape
    p2 = X2.shape[0]
    p = p1 * p2

    Delta = ECDM(X1=X1, X2=X2)

    X1c = (X1.T - X1.mean(axis=1)).T
    X2c = (X2.T - X2.mean(axis=1)).T
    sample_cross_cov = X1c @ X2c.T / max(n - 1, 1)
    s_ast = sample_cross_cov.reshape(-1, order="F")
    sparse_vec = np.zeros(p)
    order = np.argsort(np.abs(s_ast))[::-1]
    cumulative = 0.0

    for r in range(p):
        idx = order[r]
        cumulative += float(s_ast[idx] ** 2)
        sparse_vec[idx] = s_ast[idx]
        if cumulative >= Delta:
            break

    sparse_mat = sparse_vec.reshape(p1, p2, order="F")
    return sparse_mat, sample_cross_cov, float(max(Delta, 1e-12))


def sparse_cross_covariance(
    X: np.ndarray,
    Y: np.ndarray,
    random_state: Optional[int] = None,
    B: int = 200,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    del random_state, B

    X_t = _as_2d_float(X).T
    Y_t = _as_2d_float(Y).T
    Sigma_tilde, sample_cross_cov, Delta = sparse_cross_cov(X_t, Y_t)

    diagnostics = {
        "threshold_method": "ecdm_original",
        "sample_cross_cov": sample_cross_cov,
        "selected_nonzero": int(np.count_nonzero(Sigma_tilde)),
        "delta_from_ecdm": float(Delta),
    }
    return Sigma_tilde, float(max(Delta, 1e-12)), diagnostics


def cross_scale_from_matrix(Sigma_tilde_xy: np.ndarray, floor: float = 1e-12) -> float:
    scale = float(np.sum(Sigma_tilde_xy ** 2))
    return max(scale, floor)


def _cross_support_mask(Sigma_tilde_xy: np.ndarray) -> np.ndarray:
    return np.asarray(Sigma_tilde_xy != 0.0, dtype=bool)


def _estimate_high_order_moment(X: np.ndarray) -> float:
    Xc = _center(X)
    row_norm_sq = np.sum(Xc ** 2, axis=1)
    return float(np.mean(row_norm_sq ** 2))


def estimate_xi_n(
    X: np.ndarray,
    Y: np.ndarray,
    Sigma_tilde_xy: np.ndarray,
    cross_scale: float,
    alpha: float = 0.05,
    B: int = 200,
    c_max: float = 10.0,
    random_state: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    n, p1 = X.shape
    p2 = Y.shape[1]
    p = p1 + p2

    mask = _cross_support_mask(Sigma_tilde_xy)
    selected_nonzero = int(np.count_nonzero(mask))
    if selected_nonzero == 0:
        info = {
            "W1": float(_estimate_high_order_moment(X)),
            "W2": float(_estimate_high_order_moment(Y)),
            "rn_p": float(np.inf),
            "q_1_minus_alpha": float(np.inf),
            "C_hat": float(np.inf),
            "C_trunc": float(c_max),
            "T_bootstrap": np.full(B, np.inf, dtype=float),
            "cross_scale_bootstrap": np.full(B, 0.0, dtype=float),
            "selected_nonzero_reference": 0,
            "mask_based_evaluation": True,
        }
        return float(np.inf), info

    cross_scale_masked = float(np.sum(Sigma_tilde_xy[mask] ** 2))
    cross_scale_eff = max(cross_scale_masked, cross_scale, 1e-12)

    W1 = _estimate_high_order_moment(X)
    W2 = _estimate_high_order_moment(Y)
    rn_p = float(
        np.log(max(p, 2)) / max(n, 1)
        + ((W1 * W2) ** 0.25) / np.sqrt(max(n * cross_scale_eff, 1e-12))
    )

    T_vals = []
    scale_boot = []

    for _ in range(B):
        Xb, Yb = _resample_rows(X, Y, rng)
        Sigma_tilde_b, _, _ = sparse_cross_covariance(
            Xb,
            Yb,
            random_state=int(rng.integers(1_000_000)),
            B=max(10, min(50, B // 2 + 1)),
        )
        diff_b = Sigma_tilde_b[mask] - Sigma_tilde_xy[mask]
        T_b = float(np.sum(diff_b ** 2) / max(cross_scale_eff, 1e-12))
        T_vals.append(T_b)
        scale_boot.append(float(np.sum(Sigma_tilde_b[mask] ** 2)))

    T_vals_arr = np.asarray(T_vals, dtype=float)
    q = float(np.quantile(T_vals_arr, 1.0 - alpha)) if len(T_vals_arr) else 0.0
    C_hat = q / max(rn_p, 1e-12)
    C_trunc = min(C_hat, c_max)
    xi_n = float(C_trunc * rn_p)

    info = {
        "W1": float(W1),
        "W2": float(W2),
        "rn_p": float(rn_p),
        "q_1_minus_alpha": float(q),
        "C_hat": float(C_hat),
        "C_trunc": float(C_trunc),
        "T_bootstrap": T_vals_arr,
        "cross_scale_bootstrap": np.asarray(scale_boot, dtype=float),
        "selected_nonzero_reference": selected_nonzero,
        "mask_based_evaluation": True,
    }
    return xi_n, info


# ============================================================
# Feasible set and SRMR optimization
# ============================================================

def beta_scale_max_from_raw_cross(Sxy: np.ndarray, theta: SelfTheta, eps: float = 1e-12) -> float:
    raw_cross_fro_sq = float(np.sum(np.asarray(Sxy, dtype=float) ** 2))
    norm_lx_sq = float(np.sum(theta.lambda_x ** 2))
    norm_ly_sq = float(np.sum(theta.lambda_y ** 2))
    denom = max(norm_lx_sq * norm_ly_sq, eps)
    return float(np.sqrt(max(raw_cross_fro_sq, 0.0) / denom))



def beta_projection(
    Sigma_tilde_xy: np.ndarray,
    theta: SelfTheta,
    Sxy: np.ndarray,
    ridge_beta: float = 0.0,
) -> float:
    A = np.outer(theta.lambda_x, theta.lambda_y)
    mask = _cross_support_mask(Sigma_tilde_xy)
    if not np.any(mask):
        return 0.0
    denom = float(np.sum(A[mask] ** 2))
    if denom <= 1e-12:
        return 0.0
    beta_raw = float(np.sum(Sigma_tilde_xy[mask] * A[mask]) / max(denom + ridge_beta, 1e-12))
    beta_max = beta_scale_max_from_raw_cross(Sxy, theta)
    return float(np.clip(beta_raw, -beta_max, beta_max))



def cross_ratio(
    Sigma_tilde_xy: np.ndarray,
    theta: SelfTheta,
    beta: float,
    cross_scale: float,
) -> float:
    mask = _cross_support_mask(Sigma_tilde_xy)
    if not np.any(mask):
        return np.inf
    resid = Sigma_tilde_xy[mask] - beta * np.outer(theta.lambda_x, theta.lambda_y)[mask]
    scale = float(np.sum(Sigma_tilde_xy[mask] ** 2))
    scale = max(scale, cross_scale, 1e-12)
    return float(np.sum(resid ** 2) / scale)



def feasible_beta_interval(
    Sigma_tilde_xy: np.ndarray,
    theta: SelfTheta,
    Sxy: np.ndarray,
    cross_scale: float,
    xi_n: float,
) -> Optional[Tuple[float, float]]:
    A = np.outer(theta.lambda_x, theta.lambda_y)
    mask = _cross_support_mask(Sigma_tilde_xy)
    if not np.any(mask):
        return None
    A_mask = A[mask]
    S_mask = Sigma_tilde_xy[mask]
    a = float(np.sum(A_mask ** 2))
    b = -2.0 * float(np.sum(S_mask * A_mask))
    scale = max(float(np.sum(S_mask ** 2)), cross_scale, 1e-12)
    c = float(np.sum(S_mask ** 2) - xi_n * scale)
    disc = b * b - 4.0 * a * c
    if a <= 1e-12 or disc < 0:
        interval = None
    else:
        root = np.sqrt(max(disc, 0.0))
        lo = float(min((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        hi = float(max((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        interval = (lo, hi)

    beta_max = beta_scale_max_from_raw_cross(Sxy, theta)
    constraint_interval = (-beta_max, beta_max)

    if interval is None:
        return constraint_interval

    lo = max(interval[0], constraint_interval[0])
    hi = min(interval[1], constraint_interval[1])
    if lo > hi:
        return None
    return (float(lo), float(hi))


def srmr(S: np.ndarray, Sigma: np.ndarray) -> float:
    p = S.shape[0]
    num = 0.0
    diagS = np.clip(np.diag(S), 1e-12, None)
    for i in range(p):
        for j in range(i, p):
            denom = np.sqrt(diagS[i] * diagS[j])
            resid = (S[i, j] - Sigma[i, j]) / denom
            num += resid ** 2
    return float(np.sqrt((2.0 / (p * (p + 1))) * num))


# ============================================================
# Main estimator
# ============================================================

def estimate_proposed_beta_exact(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    alpha: float = 0.05,
    B_self: int = 200,
    B_cross: int = 200,
    B_threshold: int = 200,
    self_n_starts: int = 10,
    self_n_jobs: int = 10,
    candidate_strategy: str = "bootstrap",
    max_candidates: int = 200,
    c_max: float = 10.0,
    ridge: float = 0.0,
    fallback_to_best_projection: bool = True,
    random_state: Optional[int] = None,
    tau_strategy: str = 'reference_then_lambda',
    tau_reference_enforce_upper: bool = True,
) -> ProposedResult:
    X = _as_2d_float(X)
    Y = _as_2d_float(Y)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")

    n, p1 = X.shape
    _, p2 = Y.shape
    if n <= max(p1, p2):
        raise ValueError("The paper's self-part theory assumes n > p1 and n > p2.")

    rng = np.random.default_rng(random_state)
    Sxx, Syy, Sxy, _ = _sample_cov_blocks(X, Y)
    S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])

    theta_hat_self, self_nll_hat, self_info = fit_self_mle(
        Sxx, Syy,
        n_starts=self_n_starts,
        random_state=int(rng.integers(1_000_000)),
        ridge=ridge,
        n_jobs=self_n_jobs,
        shrinkage_method=None,
        shrinkage_lambda=None,
        shrinkage_apply_when="if_needed",
        tau_strategy=tau_strategy,
        tau_reference_enforce_upper=tau_reference_enforce_upper,
        verbose=True,
    )

    Sxx_fit = np.asarray(self_info.get("Sxx_input", Sxx), dtype=float)
    Syy_fit = np.asarray(self_info.get("Syy_input", Syy), dtype=float)
    S_fit = np.block([[Sxx_fit, Sxy], [Sxy.T, Syy_fit]])

    epsilon_n, self_deltas, boot_thetas = estimate_epsilon_n(
        X, Y, theta_hat_self,
        alpha=alpha, B=B_self,
        n_starts=max(3, self_n_starts // 2),
        random_state=int(rng.integers(1_000_000)),
        ridge=ridge, self_n_jobs=self_n_jobs,
        tau_strategy=tau_strategy,
        tau_reference_enforce_upper=tau_reference_enforce_upper,
    )
    epsilon_n = max(epsilon_n, 10.0)

    if candidate_strategy not in {"bootstrap", "single"}:
        raise ValueError("candidate_strategy must be 'bootstrap' or 'single'.")

    if candidate_strategy == "single":
        theta_candidates = [theta_hat_self]
    else:
        theta_candidates = [theta_hat_self]
        for theta_b in boot_thetas:
            nll_b_on_main = self_negative_loglik(theta_b, Sxx_fit, Syy_fit, ridge=ridge)
            if nll_b_on_main <= self_nll_hat + epsilon_n + 1e-12:
                theta_candidates.append(theta_b)
        unique_candidates: List[SelfTheta] = []
        signatures = set()
        for theta in theta_candidates:
            sig = tuple(np.round(np.concatenate([theta.lambda_x, theta.theta_delta, theta.lambda_y, theta.theta_epsilon, np.array([theta.tau])]), 6))
            if sig not in signatures:
                signatures.add(sig)
                unique_candidates.append(theta)
        theta_candidates = unique_candidates[:max_candidates]

    Sigma_tilde_xy, delta_hat_xy, cross_sparse_info = sparse_cross_covariance(X, Y, random_state=int(rng.integers(1_000_000)), B=B_threshold)
    cross_scale = float(np.sum(Sigma_tilde_xy ** 2))
    xi_n, xi_info = estimate_xi_n(X, Y, Sigma_tilde_xy=Sigma_tilde_xy, cross_scale=cross_scale, alpha=alpha, B=B_cross, c_max=c_max, random_state=int(rng.integers(1_000_000)))
    xi_n = max(xi_n, 10.0)

    feasible = []
    best_projection = None
    best_projection_ratio = np.inf
    for theta in theta_candidates:
        beta_hat = beta_projection(Sigma_tilde_xy, theta, Sxy)
        ratio = cross_ratio(Sigma_tilde_xy, theta, beta_hat, cross_scale)
        Sigma_hat = build_sigma_full(theta, beta_hat)
        fit = srmr(S_fit, Sigma_hat)
        interval = feasible_beta_interval(Sigma_tilde_xy, theta, Sxy, cross_scale, xi_n)
        entry = {'theta': theta, 'beta_hat': float(beta_hat), 'ratio': float(ratio), 'srmr': float(fit), 'sigma_hat': Sigma_hat, 'beta_interval': interval}
        if ratio <= xi_n + 1e-12:
            feasible.append(entry)
        if ratio < best_projection_ratio:
            best_projection_ratio = float(ratio)
            best_projection = entry

    if feasible:
        best = min(feasible, key=lambda d: d['srmr'])
        used_fallback = False
    else:
        if not fallback_to_best_projection or best_projection is None:
            raise RuntimeError('No feasible self candidate satisfied the cross constraint.')
        best = best_projection
        used_fallback = True

    diagnostics = {
        'epsilon_n': float(epsilon_n),
        'self_bootstrap_deltas': np.asarray(self_deltas, dtype=float),
        'xi_n': float(xi_n),
        'self_info': self_info,
        'cross_sparse_info': cross_sparse_info,
        'xi_info': xi_info,
        'used_fallback': bool(used_fallback),
        'candidate_strategy': candidate_strategy,
        'tau_strategy': tau_strategy,
    }
    return ProposedResult(
        beta_hat=float(best['beta_hat']),
        srmr_hat=float(best['srmr']),
        theta_hat=asdict(best['theta']),
        sigma_hat=np.asarray(best['sigma_hat'], dtype=float),
        sigma_tilde_xy=np.asarray(Sigma_tilde_xy, dtype=float),
        delta_hat_xy=float(delta_hat_xy),
        cross_scale=float(cross_scale),
        epsilon_n=float(epsilon_n),
        xi_n=float(xi_n),
        self_nll_hat=float(self_nll_hat),
        cross_ratio_hat=float(best['ratio']),
        feasible_count=int(len(feasible)),
        total_self_candidates=int(len(theta_candidates)),
        beta_interval_at_best_theta=best['beta_interval'],
        diagnostics=diagnostics,
    )


def result_to_dict(result: ProposedResult) -> Dict[str, Any]:
    return asdict(result)


# ============================================================
# Integrated unit-vector cross model
# ============================================================

@dataclass
class OneFactorUnitResult:
    u_hat: np.ndarray
    alpha_hat: float
    theta_hat: np.ndarray
    Sigma_hat: np.ndarray
    objective: float
    success: bool
    message: str


@dataclass
class UnitVectorCrossResult:
    lambda_x_hat: np.ndarray
    lambda_y_hat: np.ndarray
    beta_hat: float
    gamma_hat: float
    alpha_x_hat: float
    alpha_y_hat: float
    theta_delta_hat: np.ndarray
    theta_epsilon_hat: np.ndarray
    u_x_hat: np.ndarray
    u_y_hat: np.ndarray
    tau_used: Optional[float]
    Sxx: np.ndarray
    Syy: np.ndarray
    Sxy: np.ndarray
    self_x: Dict[str, Any]
    self_y: Dict[str, Any]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Basic utilities
# ============================================================

def _as_2d_float(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    return arr


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def sample_cov_blocks(X: ArrayLike, Y: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _as_2d_float(X)
    Y = _as_2d_float(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    Xc = _center(X)
    Yc = _center(Y)
    n = X.shape[0]
    if n < 2:
        raise ValueError("At least two observations are required.")
    denom = n - 1
    Sxx = (Xc.T @ Xc) / denom
    Syy = (Yc.T @ Yc) / denom
    Sxy = (Xc.T @ Yc) / denom
    return 0.5 * (Sxx + Sxx.T), 0.5 * (Syy + Syy.T), Sxy


def _safe_logdet_and_inv(Sigma: np.ndarray, jitter: float = 1e-8, max_tries: int = 8) -> Tuple[float, np.ndarray, float]:
    Sigma = 0.5 * (Sigma + Sigma.T)
    eye = np.eye(Sigma.shape[0])
    used = 0.0
    current = Sigma.copy()
    for k in range(max_tries):
        sign, logdet = np.linalg.slogdet(current)
        if sign > 0:
            try:
                inv = np.linalg.inv(current)
                return float(logdet), inv, used
            except np.linalg.LinAlgError:
                pass
        used = jitter * (10 ** k)
        current = Sigma + used * eye
    raise np.linalg.LinAlgError("Failed to stabilize covariance matrix.")


# ============================================================
# Self block: Sigma = alpha * u u^T + diag(theta), ||u|| = 1
# ============================================================

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm < eps:
        out = np.zeros_like(v)
        out[0] = 1.0
        return out
    return v / nrm


def _build_sigma_unit(alpha: float, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return alpha * np.outer(u, u) + np.diag(theta)


def _unpack_unit_params(z: np.ndarray, p: int) -> Tuple[np.ndarray, float, np.ndarray]:
    w = z[:p]
    u = _normalize(w)
    alpha = float(np.exp(z[p]))
    theta = np.exp(z[p + 1 : p + 1 + p])
    return u, alpha, theta


def _pack_unit_params(u: np.ndarray, alpha: float, theta: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(u, dtype=float), np.array([np.log(max(alpha, 1e-12))]), np.log(np.maximum(theta, 1e-12))])


def _unit_block_nll(z: np.ndarray, S: np.ndarray, theta_floor: float, jitter: float) -> float:
    p = S.shape[0]
    u, alpha, theta = _unpack_unit_params(z, p)
    theta = np.maximum(theta, theta_floor)
    Sigma = _build_sigma_unit(alpha, u, theta)
    logdet, inv, _ = _safe_logdet_and_inv(Sigma, jitter=jitter)
    return float(logdet + np.trace(S @ inv))


def fit_one_factor_unit_block(
    S: ArrayLike,
    n_starts: int = 10,
    random_state: Optional[int] = None,
    theta_floor: float = 1e-6,
    jitter: float = 1e-8,
    maxiter: int = 2000,
) -> OneFactorUnitResult:
    S = _as_2d_float(S)
    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be square.")
    p = S.shape[0]
    rng = np.random.default_rng(random_state)

    evals, evecs = np.linalg.eigh(S)
    u0 = _normalize(evecs[:, -1])
    if u0[0] < 0:
        u0 = -u0
    alpha0 = float(max(evals[-1], 1e-6))
    theta0 = np.maximum(np.diag(S) - alpha0 * (u0 ** 2), theta_floor)

    starts = [
        _pack_unit_params(u0, alpha0, theta0)
    ]
    for _ in range(max(0, n_starts - 1)):
        uj = _normalize(u0 + rng.normal(scale=0.25, size=p))
        if uj[0] < 0:
            uj = -uj
        alphaj = alpha0 * float(np.exp(rng.normal(scale=0.25)))
        thetaj = np.maximum(theta0 * np.exp(rng.normal(scale=0.25, size=p)), theta_floor)
        starts.append(_pack_unit_params(uj, alphaj, thetaj))

    bounds = [(None, None)] * p + [(np.log(1e-8), np.log(1e8))] + [(np.log(theta_floor), np.log(1e8))] * p

    best = None
    failures = []
    for z0 in starts:
        try:
            res = minimize(
                _unit_block_nll,
                z0,
                args=(S, theta_floor, jitter),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter},
            )
            if not np.isfinite(res.fun):
                failures.append("non-finite objective")
                continue
            if best is None or res.fun < best.fun:
                best = res
        except Exception as e:
            failures.append(f"{type(e).__name__}: {e}")

    if best is None:
        msg = failures[0] if failures else "all starts failed"
        raise RuntimeError(f"fit_one_factor_unit_block failed: {msg}")

    u_hat, alpha_hat, theta_hat = _unpack_unit_params(best.x, p)
    if u_hat[0] < 0:
        u_hat = -u_hat
    theta_hat = np.maximum(theta_hat, theta_floor)
    Sigma_hat = _build_sigma_unit(alpha_hat, u_hat, theta_hat)

    return OneFactorUnitResult(
        u_hat=u_hat,
        alpha_hat=float(alpha_hat),
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        objective=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
    )


# ============================================================
# Cross block with unit-vector directions
# ============================================================

def estimate_unit_vector_cross_model(
    X: ArrayLike,
    Y: ArrayLike,
    tau: Optional[float] = None,
    n_starts: int = 10,
    random_state: Optional[int] = None,
    theta_floor: float = 1e-6,
    jitter: float = 1e-8,
    maxiter: int = 2000,
) -> UnitVectorCrossResult:
    """
    Estimate a one-factor / one-factor model with unit-norm directions.

    Self blocks:
        Sigma_xx ~= alpha_x * u_x u_x^T + diag(theta_delta)
        Sigma_yy ~= alpha_y * u_y u_y^T + diag(theta_epsilon)
        ||u_x|| = ||u_y|| = 1

    Cross block:
        Sigma_xy ~= gamma * u_x u_y^T
    where
        gamma = <Sxy, u_x u_y^T>_F.

    If tau is supplied, reconstruct scale-aware loadings by
        c_x = sqrt(alpha_x)
        c_y = sqrt(alpha_y / tau)
        Lambda_x = c_x u_x
        Lambda_y = c_y u_y
        beta = gamma / (c_x c_y)

    If tau is not supplied, return norm-fixed loadings
        Lambda_x = u_x, Lambda_y = u_y, beta = gamma.
    """
    Sxx, Syy, Sxy = sample_cov_blocks(X, Y)

    fit_x = fit_one_factor_unit_block(
        Sxx,
        n_starts=n_starts,
        random_state=random_state,
        theta_floor=theta_floor,
        jitter=jitter,
        maxiter=maxiter,
    )
    fit_y = fit_one_factor_unit_block(
        Syy,
        n_starts=n_starts,
        random_state=None if random_state is None else random_state + 1,
        theta_floor=theta_floor,
        jitter=jitter,
        maxiter=maxiter,
    )

    ux = fit_x.u_hat
    uy = fit_y.u_hat
    rank1_dir = np.outer(ux, uy)
    gamma_hat = float(np.sum(Sxy * rank1_dir))

    if tau is None:
        lambda_x_hat = ux.copy()
        lambda_y_hat = uy.copy()
        beta_hat = gamma_hat
        tau_used = None
        c_x = 1.0
        c_y = 1.0
        mode = "norm_fixed_cross"
    else:
        tau = float(tau)
        if tau <= 0:
            raise ValueError("tau must be positive when supplied.")
        c_x = float(np.sqrt(max(fit_x.alpha_hat, 1e-12)))
        c_y = float(np.sqrt(max(fit_y.alpha_hat / tau, 1e-12)))
        lambda_x_hat = c_x * ux
        lambda_y_hat = c_y * uy
        beta_hat = float(gamma_hat / max(c_x * c_y, 1e-12))
        tau_used = tau
        mode = "tau_fixed_scale_reconstruction"

    diagnostics = {
        "mode": mode,
        "rank1_cross_projection": gamma_hat,
        "cross_fro_norm": float(np.linalg.norm(Sxy, ord="fro")),
        "direction_alignment": float(np.sum(rank1_dir * rank1_dir)),
        "c_x": float(c_x),
        "c_y": float(c_y),
        "tau_note": "When tau is None, beta absorbs the cross-block scale under unit-norm loadings.",
    }

    return UnitVectorCrossResult(
        lambda_x_hat=lambda_x_hat,
        lambda_y_hat=lambda_y_hat,
        beta_hat=float(beta_hat),
        gamma_hat=float(gamma_hat),
        alpha_x_hat=float(fit_x.alpha_hat),
        alpha_y_hat=float(fit_y.alpha_hat),
        theta_delta_hat=fit_x.theta_hat,
        theta_epsilon_hat=fit_y.theta_hat,
        u_x_hat=ux,
        u_y_hat=uy,
        tau_used=tau_used,
        Sxx=Sxx,
        Syy=Syy,
        Sxy=Sxy,
        self_x=asdict(fit_x),
        self_y=asdict(fit_y),
        diagnostics=diagnostics,
    )


# ============================================================
# Example execution
# ============================================================

def _print_result(title: str, res: UnitVectorCrossResult, beta_true: Optional[float] = None, tau_true: Optional[float] = None) -> None:
    print(f"\n=== {title} ===")
    if beta_true is not None:
        print("beta_true:", beta_true)
    print("beta_hat :", res.beta_hat)
    if tau_true is not None:
        print("tau_true :", tau_true)
    print("tau_used :", res.tau_used)
    print("gamma_hat:", res.gamma_hat)
    print("alpha_x_hat:", res.alpha_x_hat)
    print("alpha_y_hat:", res.alpha_y_hat)
    print("u_x_hat:", res.u_x_hat)
    print("u_y_hat:", res.u_y_hat)
    print("lambda_x_hat:", res.lambda_x_hat)
    print("lambda_y_hat:", res.lambda_y_hat)
    print("theta_delta_hat:", res.theta_delta_hat)
    print("theta_epsilon_hat:", res.theta_epsilon_hat)
    print("diagnostics:", res.diagnostics)


## 入力がデータの場合
def estimate_proposed_beta_exact_integrated(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    method: str = "tau_then_lambda",
    alpha: float = 0.05,
    B_self: int = 10,
    B_cross: int = 10,
    B_threshold: int = 10,
    self_n_starts: int = 10,
    self_n_jobs: int = 10,
    candidate_strategy: str = "bootstrap",
    max_candidates: int = 200,
    c_max: float = 10.0,
    ridge: float = 0.0,
    fallback_to_best_projection: bool = True,
    random_state: Optional[int] = None,
    tau_strategy: str = 'reference_then_lambda',
    tau_reference_enforce_upper: bool = True,
    theta_floor: float = 1e-6,
    jitter: float = 1e-8,
    maxiter: int = 500,
) -> Dict[str, Any]:
    """
    Unified entry point.

    method='tau_then_lambda'
        Use the original estimator from this file.

    method='unit_vector_cross'
        First estimate tau via the reference one-factor method from this file,
        then reconstruct Lambda and beta using the unit-vector cross model.
    """
    if method == 'tau_then_lambda':
        res = estimate_proposed_beta_exact(
            X, Y,
            alpha=alpha,
            B_self=B_self,
            B_cross=B_cross,
            B_threshold=B_threshold,
            self_n_starts=self_n_starts,
            self_n_jobs=self_n_jobs,
            candidate_strategy=candidate_strategy,
            max_candidates=max_candidates,
            c_max=c_max,
            ridge=ridge,
            fallback_to_best_projection=fallback_to_best_projection,
            random_state=random_state,
            tau_strategy=tau_strategy,
            tau_reference_enforce_upper=tau_reference_enforce_upper,
        )
        return {
            'method': 'tau_then_lambda',
            'result_type': 'ProposedResult',
            'result': res,
        }

    if method != 'unit_vector_cross':
        raise ValueError("method must be 'tau_then_lambda' or 'unit_vector_cross'.")

    X = _as_2d_float(X)
    Y = _as_2d_float(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must have the same number of rows.')
    n, p1 = X.shape
    _, p2 = Y.shape
    if n <= max(p1, p2):
        raise ValueError("The paper's self-part theory assumes n > p1 and n > p2.")

    Sxx, Syy, Sxy, _ = _sample_cov_blocks(X, Y)
    ref_fit = fit_one_factor_block_reference(
        Syy,
        random_state=random_state,
        maxiter=maxiter,
    )
    tau_hat = recover_tau_reference(ref_fit['alpha_hat'], ref_fit['u_hat'], Syy)
    if tau_reference_enforce_upper:
        tau_hat = min(float(tau_hat), float(ref_fit['alpha_hat']))
    tau_hat = float(max(tau_hat, 1e-8))

    uv_res = estimate_unit_vector_cross_model(
        X, Y,
        tau=tau_hat,
        n_starts=self_n_starts,
        random_state=random_state,
        theta_floor=theta_floor,
        jitter=jitter,
        maxiter=maxiter,
    )

    Sigma_xx = np.outer(uv_res.lambda_x_hat, uv_res.lambda_x_hat) + np.diag(uv_res.theta_delta_hat)
    Sigma_yy = tau_hat * np.outer(uv_res.lambda_y_hat, uv_res.lambda_y_hat) + np.diag(uv_res.theta_epsilon_hat)
    Sigma_xy = uv_res.beta_hat * np.outer(uv_res.lambda_x_hat, uv_res.lambda_y_hat)
    sigma_hat = np.block([[Sigma_xx, Sigma_xy], [Sigma_xy.T, Sigma_yy]])
    S = np.block([[Sxx, Sxy], [Sxy.T, Syy]])

    diagnostics = {
        'tau_reference': {
            'fit': ref_fit,
            'tau_hat': tau_hat,
            'strategy': 'reference_one_factor_then_unit_vector_cross',
        },
        'unit_vector': uv_res.diagnostics,
        'self_x': uv_res.self_x,
        'self_y': uv_res.self_y,
    }

    return {
        'method': 'unit_vector_cross',
        'result_type': 'UnitVectorCrossResult',
        'result': uv_res,
        'tau_hat': tau_hat,
        'sigma_hat': sigma_hat,
        'srmr_hat': srmr(S, sigma_hat),
        'diagnostics': diagnostics,
    }

# ============================================================
# Covariance-input version
# Append this block at the end of the file
# ============================================================

def _validate_cov_inputs(
    Sxx: ArrayLike,
    Syy: ArrayLike,
    Sxy: ArrayLike,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Sxy = np.asarray(Sxy, dtype=float)

    if Sxx.ndim != 2 or Sxx.shape[0] != Sxx.shape[1]:
        raise ValueError("Sxx must be a square 2D array.")
    if Syy.ndim != 2 or Syy.shape[0] != Syy.shape[1]:
        raise ValueError("Syy must be a square 2D array.")
    if Sxy.ndim != 2:
        raise ValueError("Sxy must be a 2D array.")

    p1 = Sxx.shape[0]
    p2 = Syy.shape[0]

    if Sxy.shape != (p1, p2):
        raise ValueError("Sxy must have shape (p1, p2), matching Sxx and Syy.")

    if int(n_samples) < 2:
        raise ValueError("n_samples must be at least 2.")

    Sxx = 0.5 * (Sxx + Sxx.T)
    Syy = 0.5 * (Syy + Syy.T)

    return Sxx, Syy, Sxy, int(n_samples), p1, p2


def estimate_proposed_beta_exact_from_cov(
    Sxx: ArrayLike,
    Syy: ArrayLike,
    Sxy: ArrayLike,
    n_samples: int,
    *,
    alpha: float = 0.05,
    self_n_starts: int = 10,
    self_n_jobs: int = 10,
    c_max: float = 10.0,
    ridge: float = 0.0,
    fallback_to_best_projection: bool = True,
    random_state: Optional[int] = None,
    tau_strategy: str = "reference_then_lambda",
    tau_reference_enforce_upper: bool = True,
    verbose: bool = True,
) -> ProposedResult:
    """
    Covariance-input approximation of the proposed estimator.

    Parameters
    ----------
    Sxx, Syy, Sxy : sample covariance blocks
    n_samples     : number of observations used to form the covariance matrices

    Notes
    -----
    This is NOT exactly identical to the raw-data version estimate_proposed_beta_exact(X, Y, ...),
    because epsilon_n / xi_n / thresholding in the original implementation use the raw samples.
    Here we use a covariance-only approximation:
        Sigma_tilde_xy = Sxy
        delta_hat_xy   = 0
        cross_scale    = ||Sxy||_F^2
        xi_n           = c_max
        self candidate = theta_hat_self only
    """
    Sxx, Syy, Sxy, n, p1, p2 = _validate_cov_inputs(Sxx, Syy, Sxy, n_samples)

    if n <= max(p1, p2):
        raise ValueError("The paper's self-part theory assumes n > p1 and n > p2.")

    rng = np.random.default_rng(random_state)

    S = np.block([
        [Sxx, Sxy],
        [Sxy.T, Syy],
    ])

    # --------------------------------------------------------
    # 1) self part
    # --------------------------------------------------------
    theta_hat_self, self_nll_hat, self_info = fit_self_mle(
        Sxx, Syy,
        n_starts=self_n_starts,
        random_state=int(rng.integers(1_000_000)),
        ridge=ridge,
        n_jobs=self_n_jobs,
        shrinkage_method=None,
        shrinkage_lambda=None,
        shrinkage_apply_when="if_needed",
        tau_strategy=tau_strategy,
        tau_reference_enforce_upper=tau_reference_enforce_upper,
        verbose=verbose,
    )

    # --------------------------------------------------------
    # 2) covariance-only approximation for cross part
    # --------------------------------------------------------
    sigma_tilde_xy = np.asarray(Sxy, dtype=float)
    delta_hat_xy = 0.0
    cross_scale = float(np.sum(sigma_tilde_xy ** 2))
    cross_scale = max(cross_scale, 1e-12)

    # original raw-data version estimates epsilon_n and xi_n by bootstrap/ECDM.
    # covariance-only version cannot reproduce that exactly.
    epsilon_n = 0.0
    xi_n = float(c_max)

    # --------------------------------------------------------
    # 3) choose beta on the self solution
    # --------------------------------------------------------
    interval = feasible_beta_interval(
        Sigma_tilde_xy=sigma_tilde_xy,
        theta=theta_hat_self,
        Sxy=Sxy,
        cross_scale=cross_scale,
        xi_n=xi_n,
    )

    A = np.outer(theta_hat_self.lambda_x, theta_hat_self.lambda_y)
    denom = float(np.sum(A ** 2))

    if denom <= 1e-12:
        beta_proj = 0.0
    else:
        beta_proj = float(np.sum(sigma_tilde_xy * A) / denom)

    if interval is not None:
        beta_hat = float(np.clip(beta_proj, interval[0], interval[1]))
    else:
        if not fallback_to_best_projection:
            raise RuntimeError("No feasible beta interval found in covariance-input version.")
        beta_hat = float(beta_proj)

    sigma_hat = build_sigma_full(theta_hat_self, beta_hat)
    srmr_hat = srmr(S, sigma_hat)

    resid_xy = sigma_tilde_xy - beta_hat * np.outer(theta_hat_self.lambda_x, theta_hat_self.lambda_y)
    cross_ratio_hat = float(np.sum(resid_xy ** 2) / cross_scale)

    diagnostics = {
        "input_type": "covariance_only",
        "n_samples": int(n),
        "approximation_notice": (
            "This covariance-input version is an approximation. "
            "It does not reproduce bootstrap/ECDM steps that require raw X, Y."
        ),
        "self_info": self_info,
        "used_sigma_tilde_xy": "Sxy",
        "used_delta_hat_xy": float(delta_hat_xy),
        "used_cross_scale": float(cross_scale),
        "used_epsilon_n": float(epsilon_n),
        "used_xi_n": float(xi_n),
        "beta_projection": float(beta_proj),
        "fallback_to_best_projection": bool(fallback_to_best_projection),
    }

    return ProposedResult(
        beta_hat=float(beta_hat),
        srmr_hat=float(srmr_hat),
        theta_hat=asdict(theta_hat_self),
        sigma_hat=np.asarray(sigma_hat, dtype=float),
        sigma_tilde_xy=np.asarray(sigma_tilde_xy, dtype=float),
        delta_hat_xy=float(delta_hat_xy),
        cross_scale=float(cross_scale),
        epsilon_n=float(epsilon_n),
        xi_n=float(xi_n),
        self_nll_hat=float(self_nll_hat),
        cross_ratio_hat=float(cross_ratio_hat),
        feasible_count=1,
        total_self_candidates=1,
        beta_interval_at_best_theta=interval,
        diagnostics=diagnostics,
    )


def estimate_proposed_beta_exact_integrated_from_cov(
    Sxx: ArrayLike,
    Syy: ArrayLike,
    Sxy: ArrayLike,
    n_samples: int,
    *,
    method: str = "tau_then_lambda",
    alpha: float = 0.05,
    self_n_starts: int = 10,
    self_n_jobs: int = 10,
    c_max: float = 10.0,
    ridge: float = 0.0,
    fallback_to_best_projection: bool = True,
    random_state: Optional[int] = None,
    tau_strategy: str = "reference_then_lambda",
    tau_reference_enforce_upper: bool = True,
    theta_floor: float = 1e-6,
    jitter: float = 1e-8,
    maxiter: int = 500,
) -> Dict[str, Any]:
    """
    Covariance-input unified entry point.

    method='tau_then_lambda'
        covariance-only approximation of the original proposed estimator.

    method='unit_vector_cross'
        covariance-input version of the unit-vector-cross estimator.
    """
    Sxx, Syy, Sxy, n, p1, p2 = _validate_cov_inputs(Sxx, Syy, Sxy, n_samples)

    if method == "tau_then_lambda":
        res = estimate_proposed_beta_exact_from_cov(
            Sxx, Syy, Sxy, n,
            alpha=alpha,
            self_n_starts=self_n_starts,
            self_n_jobs=self_n_jobs,
            c_max=c_max,
            ridge=ridge,
            fallback_to_best_projection=fallback_to_best_projection,
            random_state=random_state,
            tau_strategy=tau_strategy,
            tau_reference_enforce_upper=tau_reference_enforce_upper,
        )
        return {
            "method": "tau_then_lambda",
            "result_type": "ProposedResult",
            "result": res,
        }

    if method != "unit_vector_cross":
        raise ValueError("method must be 'tau_then_lambda' or 'unit_vector_cross'.")

    # --------------------------------------------------------
    # covariance-input unit_vector_cross version
    # --------------------------------------------------------
    ref_fit = fit_one_factor_block_reference(
        Syy,
        random_state=random_state,
        maxiter=maxiter,
    )
    tau_hat = recover_tau_reference(ref_fit["alpha_hat"], ref_fit["u_hat"], Syy)
    if tau_reference_enforce_upper:
        tau_hat = min(float(tau_hat), float(ref_fit["alpha_hat"]))
    tau_hat = float(max(tau_hat, 1e-8))

    uv_res = estimate_unit_vector_cross_model_from_cov(
        Sxx=Sxx,
        Syy=Syy,
        Sxy=Sxy,
        tau=tau_hat,
        n_samples=n,
        n_starts=self_n_starts,
        random_state=random_state,
        theta_floor=theta_floor,
        jitter=jitter,
        maxiter=maxiter,
    )

    return {
        "method": "unit_vector_cross",
        "result_type": "UnitVectorCrossResult",
        "result": uv_res,
    }


def estimate_unit_vector_cross_model_from_cov(
    Sxx: ArrayLike,
    Syy: ArrayLike,
    Sxy: ArrayLike,
    *,
    tau: float,
    n_samples: int,
    n_starts: int = 10,
    random_state: Optional[int] = None,
    theta_floor: float = 1e-6,
    jitter: float = 1e-8,
    maxiter: int = 500,
):
    """
    Covariance-input version corresponding to estimate_unit_vector_cross_model(X, Y, ...).

    This function assumes the original file already defines:
        - UnitVectorCrossResult
        - fit_one_factor_block_reference
        - recover_tau_reference
        - _lambda_from_direction_and_first_fixed
        - _safe_logdet_and_inv
    """
    Sxx, Syy, Sxy, n, p1, p2 = _validate_cov_inputs(Sxx, Syy, Sxy, n_samples)
    rng = np.random.default_rng(random_state)

    fit_x = fit_one_factor_block_reference(
        Sxx,
        n_starts=max(10, n_starts),
        seed=int(rng.integers(1_000_000)),
        enforce_tau_upper=True,
        maxiter=maxiter,
    )
    fit_y = fit_one_factor_block_reference(
        Syy,
        n_starts=max(10, n_starts),
        seed=int(rng.integers(1_000_000)),
        enforce_tau_upper=True,
        maxiter=maxiter,
    )

    ux = np.asarray(fit_x["u_hat"], dtype=float)
    uy = np.asarray(fit_y["u_hat"], dtype=float)

    lambda_x_hat = _lambda_from_direction_and_first_fixed(ux)
    lambda_y_hat = _lambda_from_direction_and_first_fixed(uy)

    Ax = np.outer(lambda_x_hat, lambda_y_hat)
    denom = float(np.sum(Ax ** 2))
    if denom <= 1e-12:
        beta_hat = 0.0
    else:
        beta_hat = float(np.sum(Sxy * Ax) / denom)

    gamma_hat = float(beta_hat)

    return UnitVectorCrossResult(
        lambda_x_hat=lambda_x_hat,
        lambda_y_hat=lambda_y_hat,
        beta_hat=float(beta_hat),
        gamma_hat=float(gamma_hat),
        alpha_x_hat=float(fit_x["alpha_hat"]),
        alpha_y_hat=float(fit_y["alpha_hat"]),
        theta_delta_hat=np.maximum(np.asarray(fit_x["theta_hat"], dtype=float), theta_floor),
        theta_epsilon_hat=np.maximum(np.asarray(fit_y["theta_hat"], dtype=float), theta_floor),
        u_x_hat=ux,
        u_y_hat=uy,
        tau_used=float(tau),
        Sxx=np.asarray(Sxx, dtype=float),
        Syy=np.asarray(Syy, dtype=float),
        Sxy=np.asarray(Sxy, dtype=float),
        self_x=fit_x,
        self_y=fit_y,
        diagnostics={
            "input_type": "covariance_only",
            "n_samples": int(n),
            "tau_input": float(tau),
        },
    )

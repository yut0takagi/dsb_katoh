import openpyxl as xl
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import datetime
from typing import Sequence, Literal, Dict, Any
import math


def ttest_two_groups(
    group1: Sequence[float],
    group2: Sequence[float],
    *,
    equal_var: bool | None = None,
    transform: Literal['none', 'logit'] = 'none',
    clip_eps: float = 1e-6,
    return_all: bool = True
) -> Dict[str, Any]:
    """
    2群の t 検定と効果量を計算するユーティリティ関数。
    
    Parameters
    ----------
    group1, group2 : Sequence[float]
        数値リスト（list, tuple, np.array）。None/NaN は除外。
    equal_var : bool | None
        True なら Student, False なら Welch。
        None の場合は Levene 検定 (p>=0.05) なら True, そうでなければ False を自動選択。
    transform : {'none','logit'}
        'logit' を指定すると比率(0〜1)に微小クリップした上で log(p/(1-p)) へ変換して検定。
    clip_eps : float
        logit 変換時の端点クリップ閾値。0,1 を避けるため [clip_eps, 1-clip_eps] へ収める。
    return_all : bool
        True なら補助統計（分散, 標準誤差など）も返す。
    
    Returns
    -------
    result : dict
        主キー: 
          'test' ('student' or 'welch'),
          't', 'df', 'p',
          'mean1','mean2','mean_diff',
          'ci_diff' (95%CI tuple),
          'cohens_d','hedges_g','g_ci' (95%CI),
          'glass_delta',
          'levene_p'
        transform='logit' の場合は 'inverse_means' (原スケール近似) を付加。
    """
    def _clean(x):
        arr = np.array([v for v in x if v is not None], dtype=float)
        arr = arr[~np.isnan(arr)]
        return arr

    g1 = _clean(group1)
    g2 = _clean(group2)
    if g1.size < 2 or g2.size < 2:
        raise ValueError("各群には少なくとも2つ以上の有効な数値が必要です。")

    original_means = (g1.mean(), g2.mean())

    if transform == 'logit':
        g1 = np.clip(g1, clip_eps, 1 - clip_eps)
        g2 = np.clip(g2, clip_eps, 1 - clip_eps)
        g1 = np.log(g1 / (1 - g1))
        g2 = np.log(g2 / (1 - g2))

    n1, n2 = g1.size, g2.size
    mean1, mean2 = g1.mean(), g2.mean()
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)

    # Levene で等分散性判定（center='mean'）
    levene_stat, levene_p = stats.levene(g1, g2, center='mean')

    # equal_var 自動決定
    if equal_var is None:
        equal_var = (levene_p >= 0.05)

    # t 検定
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)

    if equal_var:
        df = n1 + n2 - 2
        test_type = 'student'
        # プール分散
        sp2 = ((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2)
        se_diff = math.sqrt(sp2 * (1/n1 + 1/n2))
    else:
        # Welch の自由度
        df = (var1/n1 + var2/n2)**2 / ((var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1)))
        test_type = 'welch'
        se_diff = math.sqrt(var1/n1 + var2/n2)

    mean_diff = mean1 - mean2
    t_crit = stats.t.ppf(0.975, df)
    ci_diff = (mean_diff - t_crit*se_diff, mean_diff + t_crit*se_diff)

    # 効果量
    # プール標準偏差（等分散仮定用）
    sp = math.sqrt(((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / sp

    # Hedges g 補正
    J = 1 - 3/(4*(n1 + n2) - 9)
    hedges_g = cohens_d * J

    # g の SE (近似) & CI
    se_g = math.sqrt((n1 + n2)/(n1*n2) + (hedges_g**2)/(2*(n1 + n2 - 2)))
    g_ci = (hedges_g - t_crit*se_g, hedges_g + t_crit*se_g)

    # Glass Δ (第2群を基準：分散異質時の代替)
    glass_delta = (mean1 - mean2) / math.sqrt(var2)

    result = {
        'test': test_type,
        'equal_var_assumed': equal_var,
        't': t_stat,
        'df': df,
        'p': p_val,
        'mean1': mean1,
        'mean2': mean2,
        'mean_diff': mean_diff,
        'ci_diff': ci_diff,
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'g_ci': g_ci,
        'glass_delta': glass_delta,
        'n1': n1,
        'n2': n2,
        'var1': var1,
        'var2': var2,
        'levene_p': levene_p,
        'transform': transform,
    }

    if transform == 'logit':
        # 参考：逆 logit で平均（logit平均）を原スケールへ近似
        inv = lambda x: 1 / (1 + math.exp(-x))
        result['inverse_means'] = (inv(mean1), inv(mean2))

    if not return_all:
        # 最小限
        keys = ['test','t','df','p','mean1','mean2','mean_diff','ci_diff','hedges_g','g_ci']
        result = {k: result[k] for k in keys}
    return result

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

def run_anova_from_lists(group_data: dict, pjt:str, filename_prefix="anova_result"):
    """
    group_data: dict
        キーがグループ名、値が数値リストの辞書
        例: {
            'high': [320, 310, 300],
            'mid': [280, 270, 290],
            'low': [250, 240, 260]
        }
    filename_prefix: str
        出力ファイルのプレフィックス（デフォルト: "anova_result"）
    """

    # --- データ整形 ---
    data = []
    for group, values in group_data.items():
        for val in values:
            data.append({'group': group, 'fixation': val})

    df = pd.DataFrame(data)

    # --- 一元配置分散分析 (ANOVA) ---
    model = ols('fixation ~ group', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # --- Tukey事後比較 ---
    tukey = pairwise_tukeyhsd(endog=df['fixation'], groups=df['group'], alpha=0.05)

    # --- 結果保存 ---
    filename = f"{filename_prefix}_log.txt"  # 例: anova_result_log.txt

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"pjt: {pjt}")
        f.write(f"\n==== 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        f.write("=== 一元配置分散分析（ANOVA）結果 ===\n")
        f.write(anova_table.to_string())
        f.write("\n\n=== Tukey HSD 多重比較結果 ===\n")
        f.write(str(tukey.summary()))
        f.write("\n" + "="*50 + "\n")

    print(f"✅ 結果を保存しました: {filename}")
    return anova_table, tukey
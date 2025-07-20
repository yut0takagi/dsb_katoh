# ANOVA & Tukey HSD Analysis Project

このリポジトリは、カテゴリ群（例：アクセサリー着用頻度など）に対する一元配置分散分析（ANOVA）および事後のTukey HSD多重比較を行い、視線データなどの統計的有意性を判定するPythonプロジェクトです。

## ✨ 機能概要

* PythonでのANOVA分析 (`statsmodels`)
* Tukey HSDによる一律的な乗り越し検定
* 解析結果のログファイルへの追記保存
* Jupyter Notebook で解析プロセスを可視化

## ファイル構成

```
.
├── 1_anova_Tukey_log.txt         # ANOVA+Tukey のログ結果
├── main.ipynb                    # 解析の主ノートブック
├── main2.ipynb                   # 別パターン解析用
├── Pipfile                      # pipenv用環境定義
├── requirements.txt             # pip用依存ライブラリ
├── Project1 Metrics_notest...   # Excelの生データ
├── src_.xlsx                    # 再解析用Excel
├── t-test_rejectTrue.csv        # 有意なTukey比較のみ
├── t-test_...、高頻度_anova_Tukey_log.txt
└── ...                          # その他ログ
```

## 実行環境

* Python 3.10+
* pipenv または pip

### pipenv の場合

```bash
pipenv install
pipenv run jupyter lab
```

### pip の場合

```bash
pip install -r requirements.txt
jupyter lab
```

## 使用ライブラリ

* pandas
* scipy
* statsmodels

## 使い方

1. `main.ipynb` または `main2.ipynb` を開き、各セルを実行
2. Excel/データを読み込み解析
3. ANOVA の結果をテキストログに追記
4. Tukey HSD で比較し、有意な結果を CSV に出力

## 作成者

* takagi

## ライセンス

MIT License

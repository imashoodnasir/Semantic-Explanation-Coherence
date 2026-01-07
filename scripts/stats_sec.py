import argparse
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="csv from scripts/compute_sec.py (per-image)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="overall_sec", choices=["lesion_sec","attribute_sec","weighted_sec","overall_sec"])
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    df = df.dropna(subset=[args.metric])

    # one-way anova across xai methods
    groups = [g[args.metric].values for _, g in df.groupby("xai")]
    f_stat, p_anova = stats.f_oneway(*groups)

    # tukey hsd
    tukey = pairwise_tukeyhsd(endog=df[args.metric], groups=df["xai"], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    # add anova row
    anova_row = pd.DataFrame([{"group1":"ANOVA","group2":"","meandiff":f_stat,"p-adj":p_anova,"lower":"","upper":"","reject":p_anova < 0.05}])
    out_df = pd.concat([anova_row, tukey_df], ignore_index=True)
    out_df.to_csv(args.out, index=False)
    print("wrote:", args.out)

if __name__ == "__main__":
    main()

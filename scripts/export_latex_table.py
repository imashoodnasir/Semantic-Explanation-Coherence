import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tex = df.to_latex(index=args.index, escape=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tex)
    print("wrote:", args.out)

if __name__ == "__main__":
    main()

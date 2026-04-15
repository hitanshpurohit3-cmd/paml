from src.eda import run_eda
from src.preprocessing import preprocess
from src.monte_carlo import MonteCarlo
from src.sack import SACKClassifier

def main():
    # Step 1: EDA
    df, corr, stats = run_eda("data/parkinsons.csv")

    # Step 2: Preprocess
    X, y = preprocess(df)

    # Step 3: Model
    model = SACKClassifier()

    # Step 4: Monte Carlo Evaluation
    mc = MonteCarlo(n_splits=30)
    mc.run(model, X, y)


if __name__ == "__main__":
    main()
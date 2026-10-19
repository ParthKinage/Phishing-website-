"""Run the whole offline pipeline: data -> train -> evaluate -> importance.

Run: python -m src.run_all
"""
import time

from src import data, evaluate, importance, train


def main():
    t0 = time.perf_counter()
    print("== 1/4 data ==")
    X, y = data.load()
    data.summarise(X, y)
    print("== 2/4 train ==")
    train.main()
    print("== 3/4 evaluate ==")
    evaluate.main()
    print("== 4/4 importance ==")
    importance.main()
    print(f"Done in {time.perf_counter() - t0:.0f} s. Results in results/, models in models/.")


if __name__ == "__main__":
    main()

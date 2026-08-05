from loader import (
    main as loader_main,
)

from cleaner import (
    main as cleaner_main,
)

from eda import (
    main as eda_main,
)

from classification import (
    main as classification_main,
)

from evaluate import (
    main as evaluate_main,
)

from imbalance import (
    main as imbalance_main,
)

from model_selection import (
    main as model_selection_main,
)

from regression import (
    main as regression_main,
)

from model_persistence import (
    main as model_persistence_main,
)


def main():
    """
    Execute the complete
    analytics pipeline.
    """

    print("=" * 70)
    print("Analytics Pipeline")
    print("=" * 70)

    print("\n[1/9] Dataset Loading")
    loader_main()

    print("\n[2/9] Data Cleaning")
    cleaner_main()

    print("\n[3/9] Exploratory Data Analysis")
    eda_main()

    print("\n[4/9] Classification")
    classification_main()

    print("\n[5/9] Model Evaluation")
    evaluate_main()

    print("\n[6/9] Imbalance Handling")
    imbalance_main()

    print("\n[7/9] Hyperparameter Tuning")
    model_selection_main()

    print("\n[8/9] Regression Analysis")
    regression_main()

    print("\n[9/9] Model Persistence")
    model_persistence_main()

    print("\n" + "=" * 70)
    print("Analytics Pipeline Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
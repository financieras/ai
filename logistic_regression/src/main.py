import preprocessing
import correlations
import describe
import normalize

if __name__ == "__main__":
    preprocessing.preprocess_data()
    correlations.remove_highly_correlated_columns()
    describe.analyze_dataset()
    normalize.normalize_data()

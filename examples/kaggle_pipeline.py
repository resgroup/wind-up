import logging
import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.importance_scores = None
        self.imputer = SimpleImputer(strategy="mean")

    def mutual_information(self, X, y, threshold=0.01):
        """Select features based on mutual information with target"""
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)

        mi_scores = mutual_info_regression(X_imputed, y, random_state=self.random_state)
        self.importance_scores = pd.Series(mi_scores, index=X.columns)

        # Log feature importance details
        importance_df = pd.DataFrame({"feature": X.columns, "mutual_info_score": mi_scores}).sort_values(
            "mutual_info_score", ascending=False
        )

        logger.info("\nMutual Information Scores (top 20):")
        logger.info(importance_df.head(20))

        selected = self.importance_scores[self.importance_scores > threshold].index
        dropped_features = set(X.columns) - set(selected)
        logger.info(f"\nDropped features (mutual_info_score <= {threshold}):")
        logger.info(dropped_features)

        return X[selected]

    def model_based_selection(self, X, y, threshold="mean"):
        """Use model's feature importances for selection"""
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)

        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(X_imputed, y)

        # Log feature importance details
        importance_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        logger.info("\nRandom Forest Feature Importances (top 20):")
        logger.info(importance_df.head(20))

        selector = SelectFromModel(estimator=model, threshold=threshold)
        selector.fit(X_imputed, y)
        selected = X.columns[selector.get_support()]

        dropped_features = set(X.columns) - set(selected)
        logger.info(f"\nDropped features (below threshold '{threshold}'):")
        logger.info(dropped_features)

        return X[selected]

    def boruta_selection(self, X, y):
        """Use Boruta algorithm for feature selection"""
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)

        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        boruta = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=self.random_state)
        boruta.fit(X_imputed.values, y.values)

        # Log feature importance details
        feature_ranks = pd.DataFrame(
            {"feature": X.columns, "rank": boruta.ranking_, "selected": boruta.support_}
        ).sort_values("rank")

        logger.info("\nBoruta Feature Rankings:")
        logger.info(feature_ranks)

        selected = X.columns[boruta.support_]
        dropped_features = set(X.columns) - set(selected)
        logger.info("\nDropped features (not confirmed by Boruta):")
        logger.info(dropped_features)

        return X[selected]


class KaggleSubmissionPipeline:
    def __init__(self, model=None, tune_hyperparameters=True):
        self.tune_hyperparameters = tune_hyperparameters
        if model is not None:
            self.model = model
        elif tune_hyperparameters:
            self.model = self._get_tuned_model()
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=1000,
                learning_rate=0.1,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=1,
            )
        self.cv_scores = []
        self.feature_selector = FeatureSelector()

    def _get_tuned_model(self):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        param_distributions = {
            "learning_rate": uniform(0.01, 0.09),
            "max_depth": randint(5, 20),
            "min_samples_leaf": randint(10, 50),
            "l2_regularization": uniform(0.5, 4.5),
        }

        base_model = HistGradientBoostingRegressor(
            max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42, verbose=0
        )

        return RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=20,
            cv=10,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=2,
        )

    def evaluate_feature_selection_methods(self, X_train, y_train, cv=10):
        """Compare different feature selection methods"""
        results = {}
        methods = {
            "all_features": lambda x, y: x,
            "mutual_info": self.feature_selector.mutual_information,
            "model_based": self.feature_selector.model_based_selection,
            "boruta": self.feature_selector.boruta_selection,
        }

        for name, method in methods.items():
            logger.info(f"\nEvaluating {name}...")
            try:
                X_selected = method(X_train.copy(), y_train)
                if self.tune_hyperparameters and hasattr(self.model, "fit"):
                    logger.info("Performing hyperparameter tuning...")
                    self.model.fit(X_selected, y_train)
                    logger.info("Best parameters found:")
                    logger.info(self.model.best_params_)
                    logger.info(f"Best cross-validation MAE: {-self.model.best_score_:.4f}")

                scores = self._get_cv_scores(X_selected, y_train, cv)
                results[name] = {
                    "n_features": X_selected.shape[1],
                    "mean_mae": np.mean(scores),
                    "std_mae": np.std(scores),
                    "selected_features": list(X_selected.columns),
                }
                logger.info(f"Number of features: {X_selected.shape[1]}")
                logger.info(f"Mean MAE: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
            except Exception as e:
                logger.error(f"Error with {name}: {e!s}")

        return results

    def _get_cv_scores(self, X, y, cv):
        """Helper function to get cross-validation scores"""
        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        return -scores

    def select_features(self, X_train, y_train, X_test, method="model_based"):
        """Apply selected feature selection method to both train and test"""
        if method == "mutual_info":
            X_train_selected = self.feature_selector.mutual_information(X_train, y_train)
        elif method == "model_based":
            X_train_selected = self.feature_selector.model_based_selection(X_train, y_train)
        elif method == "boruta":
            X_train_selected = self.feature_selector.boruta_selection(X_train, y_train)
        else:
            return X_train, X_test

        selected_features = X_train_selected.columns
        X_test_selected = X_test[selected_features]
        return X_train_selected, X_test_selected

    def validate_model(self, X_train, y_train, cv=10):
        """Perform cross-validation and print results"""
        scores = self._get_cv_scores(X_train, y_train, cv)
        self.cv_scores = scores
        logger.info(f"Cross-validation MAE scores: {scores}")
        logger.info(f"Mean MAE: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")

    def train_and_predict(self, X_train, y_train, X_test):
        """Train model and generate predictions"""
        logger.info("Training final model...")

        if self.tune_hyperparameters and hasattr(self.model, "best_estimator_"):
            logger.info("Using best model from hyperparameter tuning...")
            # Get the best model from tuning
            best_model = self.model.best_estimator_
            # Fit it on the full training data
            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)

            # Log best parameters for reference
            logger.info("Best parameters used:")
            logger.info(self.model.best_params_)
        else:
            # If not tuning hyperparameters, use the base model
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            # Calculate feature importances if available
            if hasattr(self.model, "feature_importances_"):
                importances = pd.DataFrame(
                    {"feature": X_train.columns, "importance": self.model.feature_importances_}
                ).sort_values("importance", ascending=False)
                logger.info("\nTop 10 most important features:")
                logger.info(importances.head(10))

        return predictions

    def create_submission(self, predictions, sample_submission_path, output_path):
        """Create submission file"""
        submission = pd.read_csv(sample_submission_path)
        submission.iloc[:, 1] = predictions
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")


def prepare_submission(
    X_train,
    y_train,
    X_test,
    sample_submission_path,
    output_path="submission.csv",
    model=None,
    evaluate_features=True,
    feature_method="model_based",
    tune_hyperparameters=True,
):
    """Complete pipeline with feature selection and hyperparameter tuning"""
    # Remove rows where target is NaN
    mask = ~y_train.isna()
    X_train = X_train[mask]
    y_train = y_train[mask]

    pipeline = KaggleSubmissionPipeline(model, tune_hyperparameters=tune_hyperparameters)

    logger.info("Dataset information:")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info("\nMissing values in training data:")
    logger.info(X_train.isnull().sum().sort_values(ascending=False).head())

    if evaluate_features:
        logger.info("\nEvaluating feature selection methods...")
        results = pipeline.evaluate_feature_selection_methods(X_train, y_train)

    # Select features using chosen method
    X_train_selected, X_test_selected = pipeline.select_features(X_train, y_train, X_test, method=feature_method)
    logger.info(f"\nSelected {X_train_selected.shape[1]} features using {feature_method}")

    # Validate model with selected features
    pipeline.validate_model(X_train_selected, y_train)

    # Generate predictions
    predictions = pipeline.train_and_predict(X_train_selected, y_train, X_test_selected)

    # Create submission file
    pipeline.create_submission(predictions, sample_submission_path, output_path)

    return pipeline

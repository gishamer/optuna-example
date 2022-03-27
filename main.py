from ast import literal_eval

import optuna
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def create_pipeline(trial) -> Pipeline:
    pipeline = list()

    if trial.suggest_categorical('vectorizer', ['tfidf', 'count']) == 'tfidf':
        pipeline.extend([
            ('vectorizer', TfidfVectorizer(
                ngram_range=literal_eval(trial.suggest_categorical('ngram_range', ['(1, 1)', '(1, 2)', '(2, 2)'])),
                stop_words=trial.suggest_categorical('stop_words', ['english', None]),
                sublinear_tf=trial.suggest_categorical('sublinear_tf', [True, False])
            ))])
    else:
        pipeline.extend([
            ('vectorizer', CountVectorizer(
                ngram_range=literal_eval(trial.suggest_categorical('ngram_range', ['(1, 1)', '(1, 2)', '(2, 2)'])),
                stop_words=trial.suggest_categorical('stop_words', ['english', None])
            ))])

    pipeline.extend([
        ('classifier', SVC(
            max_iter=10_000,
            C=trial.suggest_float('svc_c', low=2e-5, high=2e15),
            kernel=trial.suggest_categorical('kernel', ['rbf', 'linear']),
            gamma=trial.suggest_float('svc_gamma', low=2e-15, high=2e-3)))])

    return Pipeline(pipeline)


def svm_objective(df: pd.DataFrame, trial: optuna.Trial) -> float:
    classifier_obj = create_pipeline(trial)

    skf = StratifiedKFold(n_splits=5)

    scores = cross_val_score(
        classifier_obj, df['text'], df['label'], cv=skf, scoring='f1'
    )

    return scores.mean()


def run_experiment(study: optuna.Study, df: pd.DataFrame):
    study.optimize(lambda trial: svm_objective(
        df,
        trial
    ), n_trials=50)


def evaluate_results(study_name: str):
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    best_values = study.best_trial.params
    param_importances = optuna.visualization.plot_param_importances(study)

    print(best_values)
    param_importances.show()


if __name__ == '__main__':
    study_name = 'scikit-learn-study'
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')

    newsgroups = fetch_20newsgroups(
        categories=['soc.religion.christian', 'alt.atheism'],
        remove=('headers', 'footers', 'quotes')) # remove headers and footers, since they contain information related to the category used
    data_df = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})

    run_experiment(study, data_df)

    evaluate_results(study_name)

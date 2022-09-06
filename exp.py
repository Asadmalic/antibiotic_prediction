
from ai4water.experiments import MLClassificationExperiments
from ai4water.utils.utils import dateandtime_now

from SeqMetrics import ClassificationMetrics

from load_data import data_prep


run_type = "dry_run"
target = "Meropenem_S-vs-R"
data = data_prep(target)

def f1_score(t,p)->float:
    return ClassificationMetrics(t, p).f1_score(average="macro")

def precision(t,p)->float:
    return ClassificationMetrics(t, p).precision(average="macro")


def recall(t,p)->float:
    return ClassificationMetrics(t, p).recall(average="macro")

def specificity(t,p)->float:
    return ClassificationMetrics(t, p).specificity(average="macro")


def sensitivity(t,p)->float:
    return ClassificationMetrics(t, p).sensitivity(average="macro")


comparisons = MLClassificationExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    #cross_validator={'KFold': {'n_splits': 5}},
    verbosity=0,
    train_fraction=0.75,
    val_fraction=0.3,
    monitor=[f1_score, "accuracy", precision, recall, "balanced_accuracy", sensitivity, specificity],
    exp_name = f"MLClassificationExperiments_{dateandtime_now()}_{target}_{run_type}"
)



monitor = [f1_score, "accuracy"]


comparisons.fit(data=data,
                run_type=run_type,
                opt_method = "tpe",
                num_iterations = 50,
                include=[
                    'RidgeClassifier',
                    'KNeighborsClassifier',
                    'GaussianProcessClassifier',
                    #'LinearSVC',
                    'RandomForestClassifier',
                         'XGBClassifier',
                         'GaussianProcessClassifier',
                         'HistGradientBoostingClassifier',
                         "LGBMClassifier",
                         "GradientBoostingClassifier",
                         "CatBoostClassifier",
                         "XGBRFClassifier"
                         ],)



for metric in [
    "f1_score",
    "accuracy", "precision",
               "recall",
               "balanced_accuracy", #"sensitivity",
               "specificity"
               ]:
    print(metric)
    comparisons.compare_errors(metric)

# optimize hyperparameters for best model
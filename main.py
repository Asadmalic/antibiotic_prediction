
import numpy as np

from ai4water import Model
from SeqMetrics import ClassificationMetrics
from load_data import data_prep

data = data_prep('Ciprofloxacin_S-vs-R')

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model="CatBoostClassifier",
    split_random=True,
    train_fraction=0.75,
    val_fraction=0.3,
    #x_transformation="zscore",
)

x,y = model.all_data(data=data)

# h = model.fit(data=data)
#
# t,p = model.predict_on_validation_data(data=data, return_true=True)
#
# # model.evaluate_on_training_data(data, metrics='accuracy')
# # model.evaluate_on_validation_data(data, metrics='accuracy')
# # model.evaluate_on_test_data(data, metrics='accuracy')
#
# metrics = ClassificationMetrics(t, p)
# # metrics.accuracy()
# metrics.f1_score(average='macro')

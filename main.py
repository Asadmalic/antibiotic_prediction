import matplotlib.pyplot as plt
import numpy as np

from ai4water import Model
from SeqMetrics import ClassificationMetrics
from load_data import data_prep
from utils import bar_chart

target = "Ceftazidim_S-vs-R"
data = data_prep(target)

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model="RandomForestClassifier",
    split_random=True,
    train_fraction=0.75,
    val_fraction=0.3,
    x_transformation="zscore",
)


h = model.fit(data=data)

#t,p = model.predict(data=data, return_true=True)
#
# # model.evaluate_on_training_data(data, metrics='accuracy')
# # model.evaluate_on_validation_data(data, metrics='accuracy')
# # model.evaluate_on_test_data(data, metrics='accuracy')
#
# metrics = ClassificationMetrics(t, p)
# # metrics.accuracy()
# metrics.f1_score(average='macro')


values = model._model.feature_importances_
sort_idx = np.argsort(values)
values = values[sort_idx]
labels = np.array(model.input_features)[sort_idx]

fig, ax = plt.subplots(figsize=(5, 10))
bar_chart(
    values[-30:],
    labels=labels[-30:],
    sort=True,
    show=False,
    ax=ax,
    ax_kws={"title": target},
    cmap="GnBu"
)
plt.tight_layout()
plt.show()

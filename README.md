
## Lifelike

Simple neural network approach to predicting survival curves based on maximizing the likelihood. See introduction blog article [Non-parametric survival function prediction
](https://dataorigami.net/blogs/napkin-folding/non-parametric-survival-function-prediction).


```python
from jax.experimental.stax import Dense, Dropout, Tanh
from jax.experimental import optimizers

import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import ModelCheckpoint, Logger
from lifelike.utils import dump, load


model = Model([
    Dense(20), Tanh,
    Dense(16), Tanh,
    Dropout(),
    Dense(10),
])


model.compile(optimizer=optimizers.adam,
              loss=losses.NonParametric(),
              weight_l2=0.1, smoothing_l2=10.0)

model.fit(x_train, t_train, e_train,
    epochs=1000,
    batch_size=32,
    callbacks=[ModelCheckpoint("filename.pickle"), Logger()]
)

model.predict(x_novel)


# serialization
dump(model, "filename.pickle")
model = load("filename.pickle")
model.fit(...)
```
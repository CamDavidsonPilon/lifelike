
## Lifelike

Simple neural network approach to predicting survival curves based on maximizing the likelihood.



```python
from jax.experimental.stax import Dense, Dropout, Tanh
from jax.experiment import optimizers

import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import ModelCheckpoint, Logger
from lifelike.utils import dump, load


model = Model([
    Dense(16), Tanh,
    Dense(16), Tanh,
    Dropout(),
    Dense(5),
])


model.compile(optimizer=optimizers.adam,
              loss=losses.GeneralizedGamma(),
              l2=0.1)

model.fit(x_train, t_train, e_train,
    epochs=10,
    batch_size=32,
    callbacks=[ModelCheckpoint("filename.pickle"), Logger()]
)

loss_ = model.evaluate(x_test, t_test, e_test)

model.predict(x_novel)


# serialization
dump(model, "filename.pickle")
model = load("filename.pickle")
```
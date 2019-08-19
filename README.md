
## Lifelike

Simple neural network approach to predicting survival curves based on maximizing the likelihood.



```python
from jax.experimental.stax import Dense, Dropout, Tanh
from jax.experiment import optimizers

import lifelike.losses as losses
from lifelike import Model
from lifelike.callbacks import History, ModelCheckpoint, EarlyStopping
from lifelike.utils import save


model = Model([
    Dense(16), Tanh,
    Dense(16), Tanh,
    Dropout(),
    Dense(5),
])


model.compile(optimizer=optimizers.adam,
              loss=losses.GeneralizedGamma())

model.fit(x_train, t_train, e_train,
    epochs=10,
    batch_size=32,
    callbacks=[History(), ModelCheckpoint(), EarlyStopping()]
)

loss_ = model.evaluate(x_test, t_test, e_test)

model.predict(x_novel)
save(model, "filename")
```
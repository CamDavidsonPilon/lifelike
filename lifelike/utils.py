from functools import wraps
import dill

def must_be_compiled_first(function):
    @wraps(function)
    def f(self, *args, **kwargs):
        if not self.is_compiled:
            raise ValueError("You must run .compile first on the model before using this method.")
        return function(self, *args, **kwargs)

    return f


def save(model, filepath):
    dill.dump(model, filepath)



def load(filepath):
    dill.load(filepath)
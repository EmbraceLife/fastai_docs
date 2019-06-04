from local.test import *
from local.imports import *
from local.notebook.showdoc import show_doc


def custom_dir(c, add:List):
    "Implement custom `__dir__`, adding `add` to `cls`"
    return dir(type(c)) + list(c.__dict__.keys()) + add


class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _xtra=[]
    def __getattr__(self,k):
        assert self._xtra, "Inherited from `GetAttr` but no `_xtra` attrs listed"
        if k in self._xtra: return getattr(self.default, k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self, self._xtra)
# %%
class _C(GetAttr): default,_xtra = 'Hi',['lower']

t = _C()
test_eq(t.lower(), 'hi')
test_fail(lambda: t.upper())
assert 'lower' in dir(t)

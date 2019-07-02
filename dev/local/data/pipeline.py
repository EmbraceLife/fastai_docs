#AUTOGENERATED! DO NOT EDIT! File to edit: dev/02_data_pipeline.ipynb (unless otherwise specified).

__all__ = ['get_func', 'show_title', 'Func', 'Sig', 'SelfFunc', 'Self', 'positional_annotations', 'noop_tfm',
           'PrePostInitMultiMeta', 'Transform', 'transform', 'compose_tfms', 'Pipeline', 'get_samples', 'TfmdList',
           'TfmdDS']

from ..imports import *
from ..test import *
from ..core import *
from ..notebook.showdoc import show_doc

def get_func(t, name, *args, **kwargs):
    "Get the `t.name` (potentially partial-ized with `args` and `kwargs`) or `noop` if not defined"
    f = getattr(t, name, noop)
    return f if not (args or kwargs) else partial(f, *args, **kwargs)

def show_title(o, ax=None, ctx=None):
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    ax = ifnone(ax,ctx)
    if ax is None: print(o)
    else: ax.set_title(o)

class Func():
    "Basic wrapper around a `name` with `args` and `kwargs` to call on a given type"
    def __init__(self, name, *args, **kwargs): self.name,self.args,self.kwargs = name,args,kwargs
    def __repr__(self): return f'sig: {self.name}({self.args}, {self.kwargs})'
    def _get(self, t): return get_func(t, self.name, *self.args, **self.kwargs)
    def __call__(self,t): return L(t).mapped(self._get) if is_listy(t) else self._get(t)

class _Sig():
    def __getattr__(self,k):
        def _inner(*args, **kwargs): return Func(k, *args, **kwargs)
        return _inner

Sig = _Sig()

class SelfFunc():
    "Search for `name` attribute and call it with `args` and `kwargs` on any object it's passed."
    def __init__(self, nm, *args, **kwargs): self.nm,self.args,self.kwargs = nm,args,kwargs
    def __repr__(self): return f'self: {self.nm}({self.args}, {self.kwargs})'
    def __call__(self, o):
        if not is_listy(o): return getattr(o,self.nm)(*self.args, **self.kwargs)
        else: return [getattr(o_,self.nm)(*self.args, **self.kwargs) for o_ in o]

class _SelfFunc():
    def __getattr__(self,k):
        def _inner(*args, **kwargs): return SelfFunc(k, *args, **kwargs)
        return _inner

Self = _SelfFunc()

def positional_annotations(f):
    "Get list of annotated types for all positional params, or None if no annotation"
    sig = inspect.signature(f)
    return [p.annotation if p.annotation != inspect._empty else None
            for p in sig.parameters.values() if p.default == inspect._empty and p.kind != inspect._VAR_KEYWORD]

from multimethod import multimeta,DispatchError

def _get_ret(func):
    "Get the return annotation of `func`"
    ann = getattr(func,'__annotations__', None)
    if not ann: return None
    typ = ann.get('return')
    return list(typ.__args__) if getattr(typ, '_name', '')=='Tuple' else typ

def noop_tfm(x,*args,**kwargs): return (x,*args) if len(args) > 0 else x

class PrePostInitMultiMeta(multimeta):
    "Like `PrePostInit` but inherits `multimeta`"
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        def _pass(self, *args,**kwargs): pass
        for o in ('__init__', '__pre_init__', '__post_init__'):
            if not hasattr(x,o): setattr(x,o,_pass)
        old_init = x.__init__

        @functools.wraps(old_init)
        def _init(self,*args,**kwargs):
            self.__pre_init__()
            old_init(self, *args,**kwargs)
            self.__post_init__()
        setattr(x, '__init__', _init)
        return x

class Transform(metaclass=PrePostInitMultiMeta):
    "A function that `encodes` if `filt` matches, and optionally `decodes`"
    order,add_before_setup,filt,t,state_args = 0,False,None,None,None
    def __init__(self,encodes=None,decodes=None):
        self.encodes = getattr(self, 'encodes', noop_tfm) if encodes is None else encodes
        self.decodes = getattr(self, 'decodes', noop_tfm) if decodes is None else decodes
        self.t = None

    def __post_init__(self):
        self.encodes = getattr(self, 'encodes', noop_tfm)
        self.decodes = getattr(self, 'decodes', noop_tfm)

    def _apply(self, fs, x, filt):
        if self.filt is not None and self.filt!=filt: return x
        if self.t:
            gs = self._get_func(fs, self.t)
            if is_listy(self.t) and len(positional_annotations(gs)) != len(self.t):
                gs = [self._get_func(fs,t_) for t_ in self.t]
                if len(gs) == 1: gs = gs[0]
        else: gs=fs
        if is_listy(gs): return tuple(f(x_) for f,x_ in zip(gs,x))
        return gs(*L(x)) if is_listy(self.t) else gs(x)

    def _get_func(self,f,t,ret_partial=True):
        if not hasattr(f,'__func__'): return f
        idx = (object,) + tuple(t) if is_listy(t) else (object,t)
        try: f = f.__func__[idx]
        except DispatchError: return noop_tfm
        return partial(f,self) if ret_partial else f

    def accept_types(self, t): self.t = t
        # We can't create encodes/decodes here since patching might change things later
        # So we call _get_func in _apply instead

    def return_type(self):
        g = self._get_func(self.encodes, self.t, False)
        if is_listy(self.t) and len(positional_annotations(g))-1 != len(self.t):
            return [_get_ret(self._get_func(self.encodes,t_,False)) or t_ for t_ in self.t]
        return _get_ret(g) or self.t

    def __call__(self, x, filt=None): return self._apply(self.encodes, x, filt)
    def decode  (self, x, filt=None): return self._apply(self.decodes, x, filt)
    def __getitem__(self, x): return self(x) # So it can be used as a `Dataset`

add_docs(Transform,
         __call__="Dispatch and apply the proper encodes to `x` if `filt` matches",
         decode="Dispatch and apply the proper decodes to `x` if `filt` matches",
         accept_types="Indicate the type of input received by the transform is `t`",
         return_type="Indicate the type of output the tranform returns, depending on `self.t`")

def transform(cls):
    "Decorator for registering a new `encodes` or `decodes` function in a tranform `cls`"
    def _inner(f):
        if   f.__name__=='encodes': cls.encodes.register(f)
        elif f.__name__=='decodes': cls.decodes.register(f)
        else: raise Exception('Function must be "encodes" or "decodes"')
    return _inner

def compose_tfms(x, tfms, func_nm='__call__', reverse=False, **kwargs):
    "Apply all `func_nm` attribute of `tfms` on `x`, maybe in `reverse` order"
    if reverse: tfms = reversed(tfms)
    for tfm in tfms: x = getattr(tfm,func_nm,noop)(x, **kwargs)
    return x

def _get_ret(func):
    "Get the return annotation of `func`"
    ann = getattr(func,'__annotations__', None)
    if not ann: return None
    typ = ann.get('return')
    return list(typ.__args__) if getattr(typ, '_name', '')=='Tuple' else typ

class Pipeline():
    "A pipeline of composed (for encode/decode) transforms, setup with types"
    def __init__(self, funcs=None, t=None):
        if isinstance(funcs, Pipeline): funcs = funcs.fs
        self.fs,self.t_show = [],None
        if len(L(funcs)) == 0: self.final_t = t
        else:
            for i,f in enumerate(L(funcs).sorted(key='order')):
                if not isinstance(f,Transform): f = Transform(f)
                f.accept_types(t)
                self.fs.append(f)
                if self.t_show is None and hasattr(t, 'show'): self.t_idx,self.t_show = i,t
                t = f.return_type()
            if self.t_show is None and hasattr(t, 'show'): self.t_idx,self.t_show = i+1,t
            self.final_t = t

    def new(self, t=None): return Pipeline(self, t)
    def __repr__(self): return f"Pipeline over {self.fs}"

    def setup(self, items=None):
        tfms,self.fs = self.fs,[]
        for t in tfms:
            if t.add_before_setup:     self.fs.append(t)
            if hasattr(t, 'setup'):    t.setup(items)
            if not t.add_before_setup: self.fs.append(t)

    def __call__(self, o, filt=None): return compose_tfms(o, self.fs, filt=filt)
    def decode  (self, i, filt=None): return compose_tfms(i, self.fs, func_nm='decode', reverse=True, filt=filt)
    #def __getitem__(self, x): return self(x)
    #def decode_at(self, idx): return self.decode(self[idx])
    #def show_at(self, idx):   return self.show(self[idx])

    def show(self, o, ctx=None, filt=None, **kwargs):
        if self.t_show is None: return self.decode(o, filt=filt)
        o = compose_tfms(o, self.fs[self.t_idx:], func_nm='decode', reverse=True, filt=filt)
        return self.t_show.show(o, ctx=ctx, **kwargs)

add_docs(Pipeline,
         __call__="Compose `__call__` of all `tfms` on `o`",
         decode="Compose `decode` of all `tfms` on `i`",
         new="Create a new `Pipeline`with the same `tfms` and a new initial `t`",
         show="Show item `o`",
         setup="Go through the transforms in order and call their potential setup on `items`")

def get_samples(b, max_rows=10):
    if isinstance(b, Tensor): return b[:max_rows]
    return zip(*L(get_samples(b_, max_rows) if not isinstance(b,Tensor) else b_[:max_rows] for b_ in b))

@docs
class TfmdList(GetAttr):
    "A `Pipeline` of `tfms` applied to a collection of `items`"
    _xtra = 'decode __call__ show'.split()

    def __init__(self, items, tfms, do_setup=True):
        self.items = L(items)
        self.default = self.tfms = Pipeline(tfms)
        if do_setup: self.setup()

    def __getitem__(self, i, filt=None):
        "Transformed item(s) at `i`"
        its = self.items[i]
        if is_iter(i):
            if not is_iter(filt): filt = L(filt for _ in i)
            return L(self.tfms(it, filt=f) for it,f in zip(its,filt))
        return self.tfms(its, filt=filt)

    def setup(self): self.tfms.setup(self)
    def subset(self, idxs): return self.__class__(self.items[idxs], self.tfms, do_setup=False)
    def decode_at(self, idx, filt=None):
        return self.decode(self.__getitem__(idx,filt=filt), filt=filt)
    def show_at(self, idx, filt=None, **kwargs):
        return self.show(self.__getitem__(idx,filt=filt), filt=filt, **kwargs)
    def __eq__(self, b): return all_equal(self, b)
    def __len__(self): return len(self.items)
    def __iter__(self): return (self[i] for i in range_of(self))
    def __repr__(self): return f"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms}"

    _docs = dict(setup="Transform setup with self",
                 decode_at="Decoded item at `idx`",
                 show_at="Show item at `idx`",
                 subset="New `TfmdList` that only includes items at `idxs`")

def _maybe_flat(t): return t[0] if len(t) == 1 else tuple(t)

class TfmdDS(TfmdList):
    def __init__(self, items, tfms=None, tuple_tfms=None, do_setup=True):
        if tfms is None: tfms = [None]
        self.tfmd_its = [TfmdList(items, t, do_setup=do_setup) for t in tfms]
        self.__post_init__(items, tuple_tfms, do_setup)

    def __post_init__(self, items, tuple_tfms, do_setup):
        #To avoid dupe code with DataSource
        self.items = items
        self.tfms = [it.tfms for it in self.tfmd_its]
        self.tuple_tfms = Pipeline(tuple_tfms, t=[it.tfms.final_t for it in self.tfmd_its])
        if do_setup: self.setup()

    def __getitem__(self, i, filt=None):
        its = _maybe_flat([it.__getitem__(i, filt=filt) for it in self.tfmd_its])
        if is_iter(i):
            if len(self.tfmd_its) > 1: its = zip(*L(its))
            if not is_iter(filt): filt = L(filt for _ in i)
            return L(self.tuple_tfms(it, filt=f) for it,f in zip(its,filt))
        return self.tuple_tfms(its, filt=filt)

    def __getattr__(self,k):
        for p in self.tfms+[self.tuple_tfms]:
            for f in p.fs:
                if k in L(f.state_args): return getattr(f, k)
        super().__getattr__(k)

    def __setstate__(self,data): self.__dict__.update(data) #For pickle issues

    def decode(self, o, filt=None):
        o = self.tuple_tfms.decode(o, filt=filt)
        if not is_iter(o): o = [o]
        return _maybe_flat([it.decode(o_, filt=filt) for o_,it in zip(o,self.tfmd_its)])

    def decode_batch(self, b, filt=None): return [self.decode(b_, filt=filt) for b_ in get_samples(b)]

    def show(self, o, ctx=None, filt=None, **kwargs):
        if self.tuple_tfms.t_show is not None: return self.tuple_tfms.show(o, ctx=ctx, filt=filt, **kwargs)
        o = self.tuple_tfms.decode(o, filt=filt)
        if not is_iter(o): o = [o]
        for o_,it in zip(o,self.tfmd_its): ctx = it.show(o_, ctx=ctx, filt=filt, **kwargs)
        return ctx

    def setup(self): self.tuple_tfms.setup(self)

    def subset(self, idxs):
        return self.__class__(self.items[idxs], self.tfms, self.tuple_tfms, do_setup=False)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms}\ntuple tfms - {self.tuple_tfms}"

add_docs(TfmdDS,
         "A `Dataset` created from raw `items` by calling each element of `tfms` on them",
         __getitem__="Call all `tfms` on `items[i]` then all `tuple_tfms` on the result",
         decode="Compose `decode` of all `tuple_tfms` then all `tfms` on `i`",
         decode_batch="`decode` all sample in a the batch `b`",
         show="Show item `o` in `ctx`",
         setup="Go through the transforms in order and call their potential setup on `items`",
         subset="New `TfmdDS` that only includes items at `idxs`")
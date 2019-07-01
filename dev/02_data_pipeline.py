
#%%
#default_exp data.pipeline


#%%
#export
from local.imports import *
from local.test import *
from local.core import *
from local.notebook.showdoc import show_doc


#%%
#hide
torch.cuda.set_device(int(os.environ.get('DEFAULT_GPU') or 0))

#%% [markdown]
# # Transforms and Pipeline
# 
# > Low-level transform pipelines
#%% [markdown]
# The classes here provide functionality for creating *partially reversible functions*, which we call `Transform`s. By "partially reversible" we mean that a transform can be `decode`d, creating a form suitable for display. This is not necessarily identical to the original form (e.g. a transform that changes a byte tensor to a float tensor does not recreate a byte tensor when decoded, since that may lose precision, and a float tensor can be displayed already.)
# 
# Classes are also provided and for composing transforms, and mapping them over collections. The following functionality is provided:
# 
# - A `Transform` is created with an `encodes` and potentially `decodes` function. 
# - `Pipeline` is a transform which composes transforms
# - `TfmdList` takes a collection and a transform, and provides an indexer (`__getitem__`) which dynamically applies the transform to the collection items.
# - `Tuplify` is a special `Trannsform` that takes a list of list of transforms or a list of `Pipeline`s, then aapplies them to the element it receives to return a tuple.
#%% [markdown]
# ## Convenience functions

#%%
#export
def get_func(t, name, *args, **kwargs):
    """
    "Get the `t.name` (potentially partial-ized with `args` and `kwargs`) or `noop` if not defined"

    why get_func(...)
    1. sometimes getting the plain method, t.name is not enough, 
    1. we want t.name with specified args, kwargs
    2. why not allow get_func(...) to do both
    """
    f = getattr(t, name, noop)
    return f if not (args or kwargs) else partial(f, *args, **kwargs)

#%% [markdown]
# This works for any kind of `t` supporting `getattr`, so a class or a module.

#%%
test_eq(get_func(operator, 'neg', 2)(), -2)
test_eq(get_func(operator.neg, '__call__')(2), -2)
test_eq(get_func(list, 'foobar')([2]), [2])
t = get_func(torch, 'zeros', dtype=torch.int64)(5)
test_eq(t.dtype, torch.int64)
a = [2,1]
get_func(list, 'sort')(a)
test_eq(a, [1,2])


#%%
#export
def show_title(o, ax=None, ctx=None):
    """
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    
    why show_title(...)
    1. if we really got an image, we can set `o` as the image's title
    1. if no image, then just print out `o`
    2. `ax` and `ctx` seem used interchangeably
    """
    ax = ifnone(ax,ctx)
    if ax is None: print(o)
    else: ax.set_title(o)


#%%
test_stdout(lambda: show_title("title"), "title")

#%% [markdown]
# ## Func -
#%% [markdown]
# Tranforms, are built with multiple-dispatch: a given function can have several methods depending on the type of the object received. This is done directly with the `multimethod` module and type-annotation in `Transofrm`, but you can also use the following class.

#%%
#export
class Func():
    """
    "Basic wrapper around a `name` with `args` and `kwargs` 
    to call on a given type"

    why Func():
    1. we can get a method easily like `math.pow`
    2. but what if we want the method to be `math.pow(x, 2)`
    3. what if we want to get a list [math.pow(x,2), torch.pow(x, 2)]
    4. how cool if we can get it by Func('pow', a=2)([math, torch])

    why __init__(self, name, *args, **kwargs)
    1. we get method name and its args, kwargs ready

    why __repr__(self)
    1. we just want to see method name and its args, kwargs

    why _get(self, t)
    1. we want to use get_func(...) to get method flexibly with args and kwargs

    why __call__(self, t)
    1. we want Func('pow', args, kwargs)(t) to get us:
        a. either t.pow with args, and kwargs
        b. or t1.pow(x, args, kwargs), t2.pow(x, args, kwargs)...
    """
    "Basic wrapper around a `name` with `args` and `kwargs` to call on a given type"
    def __init__(self, name, *args, **kwargs): 
        """
        why __init__(...)
        1. we get method name and its args, kwargs ready
        """
        self.name,self.args,self.kwargs = name,args,kwargs
    def __repr__(self): 
        """
        why __repr__(self)
        1. we just want to see method name and its args, kwargs
        """
        return f'sig: {self.name}({self.args}, {self.kwargs})'
    def _get(self, t): 
        """
        why _get(self, t)
        1. we want to use get_func(...) to get method flexibly with args and kwargs
        """
        return get_func(t, self.name, *self.args, **self.kwargs)
    def __call__(self,t): 
        """
        why __call__(self, t)
        1. we want Func('pow', args, kwargs)(t) to get us:
            a. either t.pow with args, and kwargs
            b. or t1.pow(x, args, kwargs), t2.pow(x, args, kwargs)...
        """
        return L(t).mapped(self._get) if is_listy(t) else self._get(t)

#%% [markdown]
# You can call the `Func` object on any module name or type, even a list of types. It will return the corresponding function (with a default to `noop` if nothing is found) or list of functions.

#%%
test_eq(Func('sqrt')(math), math.sqrt)
test_eq(Func('sqrt')(torch), torch.sqrt)

@patch
def powx(x:math, a): return math.pow(x,a)
@patch
def powx(x:torch, a): return torch.pow(x,a)
tst = Func('powx',a=2)([math, torch])
test_eq([f.func for f in tst], [math.powx, torch.powx])
for t in tst: test_eq(t.keywords, {'a': 2})


#%%
#export
class _Sig():
    """
    Sig = _Sig()
    `Sig` is just sugar-syntax to create a `Func` object more easily with the syntax `Sig.name(*args, **kwargs)`.

    why _Sig():
    1. because we want the use of Func(...) much easier
    1. how about use it in the following way:
        a. Sig.sqrt()(math)(4)
        b. Sig.pow()(math)(4,2)
        c. use , to allow vscode to display signiture
    """
    def __getattr__(self,k):
        def _inner(*args, **kwargs): return Func(k, *args, **kwargs)
        return _inner

Sig = _Sig()


#%%
show_doc(Sig, name="Sig")

#%% [markdown]
# `Sig` is just sugar-syntax to create a `Func` object more easily with the syntax `Sig.name(*args, **kwargs)`.

#%%
f = Sig.sqrt()
test_eq(f(math), math.sqrt)
test_eq(f(torch), torch.sqrt)


#%%
#export
class SelfFunc():
    "Search for `name` attribute and call it with `args` and `kwargs` on any object it's passed."
    def __init__(self, nm, *args, **kwargs): self.nm,self.args,self.kwargs = nm,args,kwargs
    def __repr__(self): return f'self: {self.nm}({self.args}, {self.kwargs})'
    def __call__(self, o):
        if not is_listy(o): return getattr(o,self.nm)(*self.args, **self.kwargs)
        else: return [getattr(o_,self.nm)(*self.args, **self.kwargs) for o_ in o]

#%% [markdown]
# The difference between `Func` and `SelfFunc` is that `Func` will generate a function when you call it on a type. On the other hand, `SelfFunc` is already a function and each time you call it on an object it looks for the `name` attribute and call it on `args` and `kwargs`.

#%%
tst = SelfFunc('sqrt')
x = torch.tensor([4.])
test_eq(tst(x), torch.tensor([2.]))
assert isinstance(tst(x), Tensor)


#%%
#export
class _SelfFunc():
    def __getattr__(self,k):
        def _inner(*args, **kwargs): return SelfFunc(k, *args, **kwargs)
        return _inner
    
Self = _SelfFunc()


#%%
show_doc(Self, name="Self")

#%% [markdown]
# `Self` is just syntax sugar to create a `SelfFunc` object more easily with the syntax `Self.name(*args, **kwargs)`.

#%%
f = Self.sqrt()
x = torch.tensor([4.])
test_eq(f(x), torch.tensor([2.]))
assert isinstance(f(x), Tensor)

#%% [markdown]
# ## Transform -

#%%
#export
def positional_annotations(f):
    "Get list of annotated types for all positional params, or None if no annotation"
    sig = inspect.signature(f)
    return [p.annotation if p.annotation != inspect._empty else None 
            for p in sig.parameters.values() if p.default == inspect._empty and p.kind != inspect._VAR_KEYWORD]


#%%
def f1(x, y:float): return x+y
def f2(a, b=2): return a
def f3(a:int, b:float=2): return a
test_eq(positional_annotations(f1), [None, float])
test_eq(positional_annotations(f2), [None])
test_eq(positional_annotations(f3), [int])


#%%
#export
from multimethod import multimeta,DispatchError


#%%
#export
def _get_ret(func):
    "Get the return annotation of `func`"
    ann = getattr(func,'__annotations__', None)
    if not ann: return None
    typ = ann.get('return')
    return list(typ.__args__) if getattr(typ, '_name', '')=='Tuple' else typ


#%%
#hide
def f1(x) -> float: return x
test_eq(_get_ret(f1), float)
def f2(x) -> Tuple[float,float]: return x
test_eq(_get_ret(f2), [float,float])


#%%
#export
def noop_tfm(x,*args,**kwargs): return (x,*args) if len(args) > 0 else x


#%%
test_eq(noop_tfm(1), 1)
test_eq(noop_tfm(1,2,3), (1,2,3))


#%%
#export
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


#%%
#export
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

#%% [markdown]
# In a transformation pipeline some steps need to be reversible - for instance, if you turn a string (such as *dog*) into an int (such as *1*) for modeling, then for display purposes you'll want to turn it back to a string again (e.g. when you have a prediction). In addition, you may wish to only run the transformation for a particular data subset, such as the training set.
# 
# `Transform` provides all this functionality. `filt` is some dataset index (e.g. provided by `DataSource`), and you provide `encodes` and optional `decodes` functions for your code. You can pass `encodes` and `decodes` functions directly to the constructor for quickly creating simple transforms. You can also create several `encodes` or `decodes` methods for different types of objects with proper type annotations.

#%%
tfm = Transform(operator.neg, decodes=operator.neg)
start = 4
t = tfm(start)
test_eq(t, -4)
test_eq(t, tfm[start]) #You can use a transform as a dataset
test_eq(tfm.decode(t), start)


#%%
def dummy_tfm(x:float,y:float): return [x+y,y]
tfm = Transform(dummy_tfm)
tfm.accept_types([float,float])
test_eq(tfm((2,3)), (5,3))
#tfm.accept_types([int,float]) Fails for now and needs a class with encodes
#test_eq(tfm((2,3)), (2,3))


#%%
class _FiltAddOne(Transform):
    filt=1
    def encodes(self, x): return x+1
    def decodes(self, x): return x-1

addt = _FiltAddOne()
test_eq(addt(start,filt=1), start+1)
test_eq(addt(start,filt=0), start)


#%%
class _DummyTfm(Transform):
    def __init__(self): pass #Pass init so that decodes isn't defined
    def encodes(self, x): return x+1

dt = _DummyTfm()
t = dt(start)
test_eq(t, start+1)
test_eq(dt.decode(t), t) #Decodes was still set at post init


#%%
show_doc(Transform.__call__)


#%%
show_doc(Transform.decode)


#%%
show_doc(Transform.accept_types)

#%% [markdown]
# At some point in the data-collection pipeline, your objects will be tuples (usually input,label). There are then different behaviors you might want your `Transform` to adopt such as:
# - being applied to the tuple and returning a new tuple (example: PointScaler takes image and a list of points, as it needs the image size to scale the points)
# - being applied to each part of the tuple (example: Cuda needs to be applied to each part of the tensor)
# - being applied to some parts of the tuple but not all, and even have different behavior depending on the type of those parts (example: data augmentation transforms should not be applied on labels, and have different behavior on images vs masks vs points vs bboxes)
# 
# You can control which behavior will be used with the signature of your `encodes` function. If it accepts several arguments (without defaults), then the transform will be applied on the tuple and expected to return a tuple. If your `encodes` function only accepts one argument, it will be applied on every part of the tuple. You can even control which part of the tuples with a type annotation: the tranform will only be applied to the items in the tuple that correspond to that type.
# 
# All of this is enabled the method `accept_types` that is called in the setup of a `Pipeline` (so out of the blue your transform object won't have this behavior). The `Pipeline` will analyze the type of objects (as given by the return annotation of any transform) and pass them along, wich tells the transform it will receive a given type (or a tuple of given types).

#%%
#Apply on the tuple as a whole
class _Add(Transform):
    def encodes(self, x, y): return (x+y,y)
    def decodes(self, x, y): return (x-y,y)

addt = _Add()
addt.accept_types([float,float])
t = addt((1,2))
test_eq(t, (3,2))
test_eq(addt.decode(t), (1,2))


#%%
#Apply on all part of the tuple
class _AddOne(Transform):
    def encodes(self, x): return x+1
    def decodes(self, x): return x-1

addt = _AddOne()
addt.accept_types([float,float])
t = addt((1,2))
test_eq(t, (2,3))
test_eq(addt.decode(t), (1,2))


#%%
#Apply on all integers of the tuple
#Also note that your tuples can have more than two elements
class _AddOne(Transform):
    def encodes(self, x:numbers.Integral): return x+1
    def encodes(self, x:float): return x*2
    def decodes(self, x:numbers.Integral): return x-1

addt = _AddOne()
addt.accept_types(float)
start = 1
t = addt(start)
test_eq(t, 2)
test_eq(addt.decode(t), 2)

addt.accept_types([float, int, float])
start = (1,2,3)
t = addt(start)
test_eq(t, (2,3,6))
test_eq(addt.decode(t), (2,2,6))


#%%
#export
def transform(cls):
    "Decorator for registering a new `encodes` or `decodes` function in a tranform `cls`"
    def _inner(f):
        if   f.__name__=='encodes': cls.encodes.register(f)
        elif f.__name__=='decodes': cls.decodes.register(f)
        else: raise Exception('Function must be "encodes" or "decodes"')
    return _inner


#%%
@transform(_AddOne)
def decodes(self, x:float): return x/2


#%%
t = addt(start)
test_eq(t, (2,3,6))
test_eq(addt.decode(t), start)


#%%
#hide
#test that addt pickles correctly
addt1 = pickle.loads(pickle.dumps(addt))
t = addt(start)
test_eq(t, (2,3,6))
test_eq(addt.decode(t), start)


#%%
show_doc(Transform.return_type)


#%%
#Check type is properly changed at dispatch
class _AddOne(Transform):
    def encodes(self, x:int)->str: return x+1
    def encodes(self, x:float):       return x*2
    def decodes(self, x:int):   return x-1
    def decodes(self, x:float): return x/2

tfm = _AddOne()
tfm.accept_types(float)
test_eq(tfm.return_type(), float)
tfm.accept_types(int)
test_eq(tfm.return_type(), str)
tfm.accept_types([int,float])
test_eq(tfm.return_type(), [str,float])


#%%
#Using supertype encodes/decodes, we have a hacky way, might want to simplify it.
class _AddOne(Transform):
    def encodes(self, x:numbers.Integral): return x+1
    def encodes(self, x:int): return self._get_func(self.encodes, numbers.Integral)(x)*2
    def decodes(self, x:numbers.Integral): return x-1
    def decodes(self, x:int): return self._get_func(self.decodes, numbers.Integral)(x/2)
    
tfm = _AddOne()
start = 2
tfm.accept_types(numbers.Integral)
t = tfm(start)
test_eq(t, 3)
test_eq(tfm.decode(t), start)
tfm.accept_types(int)
t = tfm(start)
test_eq(t, 6)
test_eq(tfm.decode(t), start)

#%% [markdown]
# ## Pipeline -

#%%
#export
def compose_tfms(x, tfms, func_nm='__call__', reverse=False, **kwargs):
    "Apply all `func_nm` attribute of `tfms` on `x`, maybe in `reverse` order"
    if reverse: tfms = reversed(tfms)
    for tfm in tfms: x = getattr(tfm,func_nm,noop)(x, **kwargs)
    return x


#%%
class _AddOne(Transform):
    def encodes(self, x): return x+1
    def decodes(self, x): return x-1
    
tfms = [_AddOne(), Transform(torch.sqrt)]
t = compose_tfms(tensor([3.]), tfms)
test_eq(t, tensor([2.]))
test_eq(compose_tfms(t, tfms, 'decodes'), tensor([1.]))
test_eq(compose_tfms(tensor([4.]), tfms, reverse=True), tensor([3.]))


#%%
#export
def _get_ret(func):
    "Get the return annotation of `func`"
    ann = getattr(func,'__annotations__', None)
    if not ann: return None
    typ = ann.get('return')
    return list(typ.__args__) if getattr(typ, '_name', '')=='Tuple' else typ


#%%
#hide
def f1(x) -> float: return x
test_eq(_get_ret(f1), float)
def f2(x) -> Tuple[float,float]: return x
test_eq(_get_ret(f2), [float,float])


#%%
#export
class Pipeline():
    "A pipeline of composed (for encode/decode) transforms, setup with types"
    def __init__(self, funcs=None, t=None): 
        if isinstance(funcs, Pipeline): funcs = funcs.raws
        self.raws,self.fs,self.t_show = L(funcs),[],None
        if len(self.raws) == 0: self.final_t = t
        else:
            for i,f in enumerate(self.raws.sorted(key='order')):
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
        tfms,raws,self.fs,self.raws = self.fs,self.raws,[],[]
        for t,r in zip(tfms,raws.sorted(key='order')):
            if t.add_before_setup:     self.fs.append(t) ; self.raws.append(r)
            if hasattr(t, 'setup'):    t.setup(items)
            if not t.add_before_setup: self.fs.append(t) ; self.raws.append(r)
                
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

#%% [markdown]
# A list of transforms are often applied in a particular order, and decoded by applying in the reverse order. `Pipeline` provides this functionality, and also ensures during its initialization that each transform get the proper functions according to the type of the previous transform. If any transform provides a type with a return annotation, this type is passed along to the next tranforms (until being overwritten by a new return annotation). Such a type can be useful when transforms filter depending on a given type (usually for data augmentation) or to provide a show method.
# 
# Here's some simple examples:

#%%
# Empty pipeline is noop
pipe = Pipeline()
test_eq(pipe(1), 1)


#%%
# Check a standard pipeline
class String():
    @staticmethod
    def show(o, ctx=None, **kwargs): return show_title(str(o), ctx=ctx)
    
class floatTfm(Transform):
    def encodes(self, x): return float(x)
    def decodes(self, x): return int(x)

float_tfm=floatTfm()
def neg(x) -> String: return -x
neg_tfm = Transform(neg, neg)
    
pipe = Pipeline([neg_tfm, float_tfm])

start = 2
t = pipe(2)
test_eq(t, -2.0)
test_eq(type(t), float)
#test_eq(t, pipe[2])
test_eq(pipe.decode(t), start)
#show decodes up to the point of the first transform that introduced the type that shows, not included
test_stdout(lambda:pipe.show(t), '-2')


#%%
# Check opposite order
pipe = Pipeline([float_tfm,neg_tfm])
t = pipe(2)
test_eq(t, -2.0)
# `show` comes from String on the last transform so nothing is decoded
test_stdout(lambda:pipe.show(t), '-2.0')


#%%
#Check filtering is properly applied
pipe = Pipeline([neg_tfm, float_tfm, _FiltAddOne()])
start = 2
test_eq(pipe(start), -2)
test_eq(pipe(start, filt=1), -1)
test_eq(pipe(start, filt=0), -2)
for t in [None, 0, 1]: test_eq(pipe.decode(pipe(start, filt=t), filt=t), start)
for t in [None, 0, 1]: test_stdout(lambda: pipe.show(pipe(start, filt=t), filt=t), "-2")


#%%
#Check type is properly changed at dispatch
class _AddOne(Transform):
    def encodes(self, x:int)->String: return x+1
    def encodes(self, x:float):       return x*2
    def decodes(self, x:int):   return x-1
    def decodes(self, x:float): return x/2

pipe = Pipeline(_AddOne(), t=int)
test_eq(pipe.final_t, String)
pipe = Pipeline(_AddOne(), t=float)
test_eq(pipe.final_t, float)
pipe = Pipeline(_AddOne(), t=[int,float])
test_eq(pipe.final_t, [String,float])

#%% [markdown]
# ### Methods

#%%
show_doc(Pipeline.__call__)


#%%
show_doc(Pipeline.decode)


#%%
show_doc(Pipeline.new)


#%%
show_doc(Pipeline.setup)

#%% [markdown]
# During the setup, the `Pipeline` starts with no transform and adds them one at a time, so that during its setup, each transfrom get the items processed up to its point and not after. Depending on the attribute `add_before_setup`, the transform is added after the setup (default behaivor) so it's not called on the items used for the setup, or before (in which case it's called on the values used for setup).

#%%
#hide
#Test is below with TfmdList

#%% [markdown]
# ## TfmedList -

#%%
#export
def get_samples(b, max_rows):
    if isinstance(b, Tensor): return b[:max_rows]
    return zip(*L(get_samples(b_, max_rows) if not isinstance(b,Tensor) else b_[:max_rows] for b_ in b))


#%%
#export
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

#%% [markdown]
# `tfms` can either be a `Pipeline` or a list of transforms.

#%%
tl = TfmdList([1,2,3], [neg_tfm, float_tfm])
t = tl[1]
test_eq(t, -2.0)
test_eq(type(t), float)
test_eq(tl.decode_at(1), 2)
test_eq(tl.decode(t), 2)
test_stdout(lambda: tl.show_at(2), '-3')
tl


#%%
p2 = tl.subset([0,2])
test_eq(p2, [-1.,-3.])

#%% [markdown]
# Here's how we can use `TfmdList.setup` to implement a simple category list, getting labels from a mock file list:

#%%
class _Cat(Transform):
    order = 1
    def __init__(self, subset_idx=None): self.subset_idx = subset_idx
    def encodes(self, o): return self.o2i[o]
    def decodes(self, o): return self.vocab[o]
    def setup(self, items): 
        if self.subset_idx is not None: items = items.subset(self.subset_idx)
        self.vocab,self.o2i = uniqueify(items, sort=True, bidir=True)

def _lbl(o) -> String: return o.split('_')[0]

test_fns = ['dog_0.jpg','cat_0.jpg','cat_2.jpg','cat_1.jpg','dog_1.jpg']
tcat = _Cat()
tl = TfmdList(test_fns, [tcat,_lbl])

test_eq(tcat.vocab, ['cat','dog'])
test_eq([1,0,0,0,1], tl)
test_eq(1, tl[-1])
test_eq([1,0], tl[0,1])
t = list(tl)
test_eq([1,0,0,0,1], t)
test_eq(['dog','cat','cat','cat','dog'], map(tl.decode,t))
test_stdout(lambda:tl.show_at(0), "dog")
tl


#%%
tcat = _Cat([0,1,2])
tl = TfmdList(test_fns, [tcat,_lbl])


#%%
#hide
#Test of add_before_setup
class _AddSome(Transform):
    def __init__(self):   self.a = 2
    def encodes(self, x): return x+self.a
    def decodes(self, x): return x-self.a
    def setup(self, items): self.a = tensor(items).float().mean().item()
        
tl1 = TfmdList([1,2,3,4], _AddSome())
test_eq(tl1.tfms.fs[0].a, 2.5) #Setup on the raw items, mean is 2.5

_AddSome.add_before_setup = True
tl1 = TfmdList([1,2,3,4], _AddSome())
test_eq(tl1.tfms.fs[0].a, 4.5) #Setup on the tfmed items, mean is 4.5


#%%
#hide
#Check filtering is properly applied
tl1 = TfmdList([1,2,3,4], [neg_tfm, float_tfm, _FiltAddOne()])
test_eq(tl1[2], -3)
test_eq(tl1.__getitem__(2, filt=1), -2)
test_eq(tl1.__getitem__(2, filt=0), -3)
test_eq(tl1.__getitem__([2,2], filt=[0,1]), [-3,-2])
for t in [None, 0, 1]: test_eq(tl1.decode(tl1.__getitem__(1, filt=t), filt=t), 2)
for t in [None, 0, 1]: test_eq(tl1.decode_at(1, filt=t), 2)
for t in [None, 0, 1]: test_stdout(lambda: tl1.show_at(1, filt=t), "-2")

#%% [markdown]
# ### Methods

#%%
show_doc(TfmdList.__getitem__)


#%%
show_doc(TfmdList.decode_at)


#%%
test_eq(tl.decode_at(1),tl.decode(tl[1]))


#%%
show_doc(TfmdList.show_at)


#%%
test_stdout(lambda: tl.show_at(1), 'cat')


#%%
show_doc(TfmdList.subset)

#%% [markdown]
# ## TfmdDS -

#%%
#exports
def _maybe_flat(t): return t[0] if len(t) == 1 else tuple(t)


#%%
#export
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


#%%
#export 
add_docs(TfmdDS,
         "A `Dataset` created from raw `items` by calling each element of `tfms` on them",
         __getitem__="Call all `tfms` on `items[i]` then all `tuple_tfms` on the result",
         decode="Compose `decode` of all `tuple_tfms` then all `tfms` on `i`",
         decode_batch="`decode` all sample in a the batch `b`",
         show="Show item `o` in `ctx`",
         setup="Go through the transforms in order and call their potential setup on `items`",
         subset="New `TfmdDS` that only includes items at `idxs`")

#%% [markdown]
# `tfms` is a list of objects that can be:
# - one transform
# - a list of transforms
# - a `Pipeline`

#%%
class _TNorm(Transform):
    state_args = ['m', 's']
    def encodes(self, o): return (o-self.m)/self.s
    def decodes(self, o): return (o*self.s)+self.m
    def setup(self, items):
        its = tensor(items).float()
        self.m,self.s = its.mean(),its.std()


#%%
items = [1,2,3,4]
#If you pass only a list with one list of tfms/Pipeline, TfmdDS behaves like TfmOver
tds = TfmdDS(items, [neg_tfm])
test_eq(tds[0], -1)
test_eq(tds[[0,1,2]], [-1,-2,-3])
test_eq(tds.decode_at(0), 1)
test_stdout(lambda:tds.show_at(1), '-2')


#%%
items = [1,2,3,4]
tds = TfmdDS(items, [neg_tfm, [neg_tfm,_TNorm()]])
x,y = zip(*tds)
test_close(tensor(y).mean(), 0)
test_close(tensor(y).std(), 1)
test_eq(x, [-1,-2,-3,-4])
test_stdout(lambda:tds.show_at(1), '-2\ntensor(-2.)')
test_eq(tds.m, tds.tfms[1].fs[1].m)
test_eq(tds.s, tds.tfms[1].fs[1].s)


#%%
#hide
#Test if show at the tuple level interrupts decoding
class DoubleString():
    @staticmethod
    def show(o, ctx=None, **kwargs): print(o[0],o[1])

class _DummyTfm(Transform):
    def encodes(self, x,y)->DoubleString: return [x,y]

items = [1,2,3,4]
tds = TfmdDS(items, [neg_tfm, neg_tfm], _DummyTfm())
test_stdout(lambda: tds.show_at(0), "-1 -1")


#%%
#hide
#Check filtering is properly applied
tds = TfmdDS(items, [neg_tfm, [neg_tfm,_FiltAddOne()]])
test_eq(tds[1], [-2,-2])
test_eq(tds.__getitem__(1, filt=1), [-2,-1])
test_eq(tds.__getitem__(1, filt=0), [-2,-2])
test_eq(tds.__getitem__([1,1], filt=[0,1]), [[-2,-2], [-2,-1]])
for t in [None, 0, 1]: test_eq(tds.decode(tds.__getitem__(1, filt=t), filt=t), [2,2])
for t in [None, 0, 1]: test_eq(tds.decode_at(1, filt=t), [2,2])
for t in [None, 0, 1]: test_stdout(lambda: tds.show_at(1, filt=t), "-2\n-2")


#%%
tds.__getitem__([1,1], filt=[0,1])


#%%
show_doc(TfmdDS.__getitem__)


#%%
show_doc(TfmdDS.decode)


#%%
show_doc(TfmdDS.decode_at)


#%%
show_doc(TfmdDS.show)


#%%
show_doc(TfmdDS.show_at)


#%%
show_doc(TfmdDS.setup)


#%%
show_doc(TfmdDS.subset)

#%% [markdown]
# ## Export -

#%%
#hide
from local.notebook.export import notebook2script
notebook2script(all_fs=True)


#%%




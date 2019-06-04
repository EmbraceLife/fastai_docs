# Fastai-v2 Docs on Each Item

- search "doc_improve:" with vim Ag to see my proposed source improvements in the source files below
- search "make_uncool:" with vim Ag to see how clean and compact official source code is and how to make it uncool for debugging
- search "not_finished": for official source but unfinished properly

[core.newchk](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.newchk.py)
> enable a class to create a new instance (normal) or return the input if the input is already an instance (new feature)
![core.newchk](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/images/core.newchk.png =100x50)

<img src="https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/images/core.newchk.png" alt="drawing" height="200" width="100"/>

[core.patch](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.patch.py)
> enable a function to add itself to the Class of its first parameter

[core.chk](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.chk.py)
> enable a function to check on its parameters types

[core.ls](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.ls.py)
> enable a Path object with a new method to check its contents on the immediate level

[core.tensor](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.tensor.py)
> put array-like, list, tuple, or just a few numbers into an tensor

[core.tensor.ndim](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.tensor.ndim.py)
> add `ndim` as a property to any tensor object to return num of dimensions

[core.add_docs](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.add_docs.py)
> to add docs for Class and methods and report which has no docs yet

[core.docs](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.docs.py)
> to enable a Class to set up its docs (unfinished by official source yet)

The practical usage of custom_dir and GetAttr
[core.custom_dir, core.GetAttr](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.getattr.py)
> 1. enable a subclass to take all its methods into its `__dir__` using `custom_dir`
> 2. access additional methods from `_xtra` using `__getattr__`

[core.is_iter](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.is_iter.py)
> to check anything is iterable or not, but Rank 0 tensors in PyTorch is not iterable

[core.coll_repr](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.coll_repr.py)
> to print out a collection under 10 items

[core._listify](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core._listify.py)
> turn everything into a list

[core._mask2idxs](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core._mask2idxs.py)
> make indexes or binary indexes

[core.L](https://github.com/EmbraceLife/fastai_docs/blob/my-v2/my-docs/core.L.py)
> the most easy-to-use and powerful list class with all the utils needed

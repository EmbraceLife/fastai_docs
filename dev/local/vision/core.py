#AUTOGENERATED! DO NOT EDIT! File to edit: dev/06_vision_core.ipynb (unless otherwise specified).

__all__ = ['Image', 'PILImage', 'Imagify', 'Mask', 'Maskify', 'TensorPoint', 'Pointify', 'TensorBBox', 'BBoxify',
           'get_annotations', 'ImageConverter', 'image_resize', 'ImageResizer', 'image2byte', 'ImageToByteTensor',
           'PointScaler', 'BBoxScaler', 'BBoxCategorize']

from ..imports import *
from ..test import *
from ..core import *
from ..data.pipeline import *
from ..data.core import *
from ..data.external import *

from PIL import Image

class PILImage():
    "Basic type for PIL Images"
    kwargs = dict(cmap='viridis')
    @staticmethod
    def show(o, ctx=None, **kwargs): return show_image(o, ctx=ctx, **{**PILImage.kwargs, **kwargs})

class Imagify(Transform):
    "Open an `Image` from path `fn`"
    def __init__(self, func=Image.open):  self.func = func
    def encodes(self, fn)->PILImage: return self.func(fn)

class Mask(PILImage):
    "Basic type for a segmentation mask as a PIL Image"
    kwargs = dict(cmap='tab20', alpha=0.5)
    @staticmethod
    def show(o, ctx=None, **kwargs): return show_image(o, ctx=ctx, **{**Mask.kwargs, **kwargs})

class Maskify(Transform):
    "Open an `Image` from path `fn` as `Mask`"
    def __init__(self, func=Image.open): self.func = func
    def encodes(self, fn)->Mask: return self.func(fn)

class TensorPoint():
    "Basic type for points in an image"
    kwargs = dict(s=10, marker='.', c='r')
    @staticmethod
    def show(o, ctx=None, **kwargs):
        if 'figsize' in kwargs: del kwargs['figsize']
        ctx.scatter(o[:, 0], o[:, 1], **{**TensorPoint.kwargs, **kwargs})
        return ctx

class Pointify(Transform):
    "Convert an array or a list of points `t` to a `Tensor`"
    def encodes(self, t)->TensorPoint: return tensor(t).view(-1, 2).float()

from fastai.vision.data import get_annotations
from matplotlib import patches, patheffects

def _draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def _draw_rect(ax, b, color='white', text=None, text_size=14, hw=True, rev=False):
    lx,ly,w,h = b
    if rev: lx,ly,w,h = ly,lx,h,w
    if not hw: w,h = w-lx,h-ly
    patch = ax.add_patch(patches.Rectangle((lx,ly), w, h, fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(lx,ly, text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)

class TensorBBox(TensorPoint):
    "Basic type for a list of bounding boxes in an image"
    @staticmethod
    def show(x, ctx=None, **kwargs):
        bbox,label = x
        for b,l in zip(bbox, label):
            if l != '#bg': _draw_rect(ctx, b, hw=False, text=l)
        return ctx

class BBoxify(Transform):
    "Convert an list of bounding boxes `t` to (`Tensor`,labels) tuple"
    def encodes(self, x)->TensorBBox: return (tensor(x[0]).view(-1, 4).float(),x[1])

def get_annotations(fname, prefix=None):
    "Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
    annot_dict = json.load(open(fname))
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    classes = {o['id']:o['name'] for o in annot_dict['categories']}
    for o in annot_dict['annotations']:
        bb = o['bbox']
        id2bboxes[o['image_id']].append([bb[0],bb[1], bb[0]+bb[2], bb[1]+bb[3]])
        id2cats[o['image_id']].append(classes[o['category_id']])
    id2images = {o['id']:ifnone(prefix, '') + o['file_name'] for o in annot_dict['images'] if o['id'] in id2bboxes}
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]

class ImageConverter(Transform):
    "Convert an image or a mask to `mode`/`mode_mask`"
    def __init__(self, mode='RGB', mask_mode='L'): self.modes = (mode,mask_mode)
    def encodes(self, o:PILImage): return o.convert(self.modes[0])
    def encodes(self, o:Mask):     return o.convert(self.modes[1])

def image_resize(img, size, resample=Image.BILINEAR):
    "Resize image to `size` using `resample"
    return img.resize((size[1],size[0]), resample=resample)
image_resize.order=10

class ImageResizer(Transform):
    order=1
    "Resize image to `size` using `resample"
    def __init__(self, size, resample=Image.BILINEAR):
        if not is_listy(size): size=(size,size)
        self.size,self.resample = size,resample

    def encodes(self, o:PILImage): return image_resize(o, size=self.size, resample=self.resample)

def image2byte(img):
    "Transform image to byte tensor in `c*h*w` dim order."
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    w,h = img.size
    return res.view(h,w,-1).permute(2,0,1)

class ImageToByteTensor(Transform):
    "Transform image to byte tensor in `c*h*w` dim order."
    order=10
    def encodes(self, o:PILImage)->TensorImage: return image2byte(o)
    def encodes(self, o:Mask)    ->TensorMask:  return image2byte(o)[0]

class PointScaler(Transform):
    "Scale a tensor representing points"
    def __init__(self, do_scale=True, y_first=False): self.do_scale,self.y_first = do_scale,y_first

    def encodes(self, x, y:TensorPoint):
        if self.y_first: y = y.flip(1)
        sz = [x.shape[-1], x.shape[-2]] if isinstance(x, Tensor) else x.size
        if self.do_scale: y = y * 2/tensor(sz).float() - 1
        return (x,y)

    def decodes(self, x, y:TensorPoint):
        sz = [x.shape[-1], x.shape[-2]] if isinstance(x, Tensor) else x.size
        y = (y+1) * tensor(sz).float()/2
        return (x,y)

class BBoxScaler(PointScaler):
    "Scale a tensor representing bounding boxes"
    def encodes(self, x, y:TensorBBox):
        scaled_bb = self._get_func(super().encodes, self.t)(x,y[0].view(-1,2))[1]
        return (x,(scaled_bb.view(-1,4),y[1]))

    def decodes(self, x, y:TensorBBox):
        scaled_bb = self._get_func(super().decodes, self.t)(x,y[0].view(-1,2))[1]
        return (x, (scaled_bb.view(-1,4), y[1]))

class BBoxCategorize(Transform):
    "Reversible transform of category string to `vocab` id"
    order,state_args=1,'vocab'
    def __init__(self, vocab=None, subset_idx=None):
        self.vocab,self.subset_idx = vocab,subset_idx
        self.o2i = None if vocab is None else {v:k for k,v in enumerate(vocab)}

    def setup(self, dsrc):
        if not dsrc: return
        dsrc = dsrc.train if self.subset_idx is None else dsrc.subset(self.subset_idx)
        vals = set()
        for b,c in dsrc: vals = vals.union(set(c))
        self.vocab,self.otoi = uniqueify(list(vals), sort=True, bidir=True, start='#bg')

    def encodes(self, o): return (o[0],tensor([self.otoi[o_] for o_ in o[1] if o_ in self.otoi]))
    def decodes(self, i): return (i[0],[self.vocab[i_] for i_ in i[1]])
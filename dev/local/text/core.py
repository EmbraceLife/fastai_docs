#AUTOGENERATED! DO NOT EDIT! File to edit: dev/30_text_core.ipynb (unless otherwise specified).

__all__ = ['parallel', 'ProcessFunc', 'parallel_by_batch', 'spec_add_spaces', 'rm_useless_spaces', 'replace_rep',
           'replace_wrep', 'fix_html', 'replace_all_caps', 'replace_maj', 'lowercase', 'BaseTokenizer',
           'SpacyTokenizer', 'apply_rules', 'tokenize1', 'TokenizeBatch', 'parallel_tokenize', 'create_folders',
           'read_text', 'tokenize_folder', 'tokenize_df', 'tokenize_csv', 'SentencePieceTokenizer']

from ..imports import *
from ..test import *
from ..core import *
from ..data.core import *
from ..data.external import *
from ..notebook.showdoc import show_doc

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Pipe
import spacy,html
from spacy.symbols import ORTH

def parallel(func, items, n_cpus=defaults.cpus):
    "Applies `func` in parallel to `items`, using `n_cpus`"
    if n_cpus<2: results = [func(o) for o in progress_bar(items, leave=False)]
    else:
        with ProcessPoolExecutor(max_workers=n_cpus) as ex:
            return [x for x in progress_bar(ex.map(func,items), total=len(items), leave=False)]

class ProcessFunc():
    "A class for functions you want executed in `parallel_by_batch`"
    def iterate(self, batch): return (b for b in batch)

def _wrap_process_func(pfunc_cls, batch, send_end, *args, start_idx=None, **kwargs):
    f = pfunc_cls(*args, **kwargs)
    if start_idx is not None:
        for i,b in enumerate(f.iterate(batch)): send_end.send((start_idx+i,b))
    else:
        for b in f.iterate(batch): send_end.send(b)

def parallel_by_batch(pfunc_cls, items, *args, n_cpus=defaults.cpus, enum_res=True, **kwargs):
    "Instantiate `pfunc_cls` in `n_cpus` process then call their iterate on batch of `items` in parallel."
    recv_end, send_end = Pipe(False)
    batches = np.array_split(items, n_cpus)
    idx = np.cumsum(0 + L(len(b) for b in batches)) if enum_res else [None] * n_cpus
    processes = [Process(target=_wrap_process_func, args=(pfunc_cls, b, send_end, *args),
                         kwargs={'start_idx':i, **kwargs}) for b,i in zip (batches, idx)]
    for p in processes: p.start()
    for _ in progress_bar(items, leave=False): yield recv_end.recv()
    for p in processes: p.join()

#special tokens
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()

_re_spec = re.compile(r'([/#\\])')

def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)

_re_space = re.compile(' {2,}')

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)

_re_rep = re.compile(r'(\S)(\1{3,})')

def replace_rep(t):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"
    def _replace_rep(m):
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    return _re_rep.sub(_replace_rep, t)

_re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+){2,})\1(\s|\W|$)')

def replace_wrep(t):
    "Replace word repetitions: word word word word -> TK_WREP 4 word"
    def _replace_wrep(m):
        c,cc,e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'
    return _re_wrep.sub(_replace_wrep, t)

def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(' @-@ ','-')
    return html.unescape(x)

_re_all_caps = re.compile(r'(\s|^)([A-Z]+[^a-z\s]*)(?=(\s|$))')

def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_all_caps.sub(_replace_all_caps, t)

_re_maj = re.compile(r'(\s|^)([A-Z][^A-Z\s]*)(?=(\s|$))')

def replace_maj(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_maj.sub(_replace_maj, t)

def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]
defaults.text_proc_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces,
                            replace_all_caps, replace_maj, lowercase]
defaults.text_token_sep = '\u2581'

class BaseTokenizer():
    "Basic tokenizer that just splits on spaces"
    def __init__(self, **kwargs): pass
    def pipe(self, items):
        for t in items: yield t.split()

class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, batch_size=5000):
        special_toks = ifnone(special_toks, defaults.text_spec_tok)
        self.nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
        for w in special_toks: self.nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.batch_size=batch_size

    def pipe(self, items):
        for doc in self.nlp.pipe(items, batch_size=self.batch_size):
            yield [d.text for d in doc]

def apply_rules(items, rules):
    "Returns a generator that apply `rules`  to `items`"
    for o in items: yield compose(*rules)(o)

def tokenize1(text, tok_func=SpacyTokenizer, rules=None, **tok_kwargs):
    "Tokenize one `text` with an instance of `tok_func` and some `rules`"
    rules = L(ifnone(rules, defaults.text_proc_rules))
    tokenizer = tok_func(**tok_kwargs)
    for tok in tokenizer.pipe(apply_rules([text], rules)): return tok

class TokenizeBatch(ProcessFunc):
    "A wrapper around `tok_func` to apply `rules` and tokenize in parallel"
    def __init__(self, tok_func, rules, **tok_kwargs ): self.tok,self.rules = tok_func(**tok_kwargs),rules
    def iterate(self, batch): return self.tok.pipe(apply_rules(batch, self.rules))

def parallel_tokenize(items, tok_func, rules, **tok_kwargs):
    "Calls a potential setup on `tok_func` before launching `TokenizeBatch` in parallel"
    if hasattr(tok_func, 'setup'): tok_kwargs = tok_func(**tok_kwargs).setup(items, rules)
    return parallel_by_batch(TokenizeBatch, items, tok_func, rules, **tok_kwargs)

def create_folders(path, output_dir, include=None):
    "Scan `path` and create the same folder architecture in `output_dir`"
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
        if include is not None and i==0: d[:] = [o for o in d if o in include]
        else:                            d[:] = [o for o in d if not o.startswith('.')]
        for x in d: os.makedirs(output_dir/(Path(p)/Path(x)).relative_to(path), exist_ok=True)

def read_text(fname):
    "Read the content of `fname`"
    with open(fname, 'r') as f: return f.read()

def tokenize_folder(path, extensions=None, include=None, output_dir=None, n_cpus=defaults.cpus,
                    rules=None, tok_func=SpacyTokenizer, **tok_kwargs):
    "Tokenize text files in `path` in parallel using `n_workers`"
    path = Path(path)
    extensions = ifnone(extensions, ['.txt'])
    fnames = get_files(path, extensions=extensions, recurse=True, include=include)
    output_dir = Path(ifnone(output_dir, path.parent/f'{path.name}_tok'))
    create_folders(path, output_dir, include=include)
    rules = read_text + L(ifnone(rules, defaults.text_proc_rules.copy()))
    counter = Counter()

    for i,tok in parallel_tokenize(fnames, tok_func, rules, **tok_kwargs):
        out = output_dir/fnames[i].relative_to(path)
        with open(out, 'w') as f: f.write(defaults.text_token_sep.join(tok))
        with open(out.parent/f'{out.stem}.len', 'w') as f: f.write(str(len(tok)))
        counter.update(Counter(tok))

    pickle.dump(counter, open(output_dir/'counter.pkl','wb'))

def _join_texts(df, mark_fields=False):
    "Join texts in row `idx` of `df`, marking each field with `FLD` if `mark_fields=True`"
    text_col = (f'{FLD} {1} ' if mark_fields else '' ) + df.iloc[:,0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df.iloc[:,i].astype(str)
    return text_col.values

def tokenize_df(df, text_cols, n_workers=defaults.cpus, rules=None, mark_fields=None,
                tok_func=SpacyTokenizer, **tok_kwargs):
    "Tokenize texts in `df[text_cols]` in parallel using `n_workers`"
    text_cols = L(text_cols)
    mark_fields = ifnone(mark_fields, len(text_cols) > 1)
    rules = L(ifnone(rules, defaults.text_proc_rules.copy()))
    texts = _join_texts(df[text_cols], mark_fields=mark_fields)
    lengths,outputs,counter = np.zeros(len(df)),np.zeros(len(df), dtype=np.object),Counter()

    for i,tok in parallel_tokenize(texts, tok_func, rules, **tok_kwargs):
        lengths[i],outputs[i] = len(tok),defaults.text_token_sep.join(tok)
        counter.update(Counter(tok))

    other_cols = [c for c in df.columns if c not in text_cols]
    res = df[other_cols].copy()
    res['text'],res['text_lengths'] = outputs,lengths
    return res, counter

#TODO: test + rework
def tokenize_csv(fname, text_cols, outname=None, n_workers=4, rules=None, mark_fields=None,
                 tok_func=SpacyTokenizer, header='infer', chunksize=None, **tok_kwargs):
    "Tokenize texts in the `text_cols` of the csv `fname` in parallel using `n_workers`"
    df = pd.read_csv(fname, header=header, chunksize=chunksize)
    outname = Path(ifnone(outname, fname.parent/f'{fname.stem}_tok.csv'))
    kwargs = dict(n_workers=n_workers, pre_rules=pre_rules, post_rules=post_rules,
                  mark_fields=mark_fields, tok_func=tok_func, **tok_kwargs)
    if chunksize is None:
        out,cnt = tok_df(df, text_cols, **kwargs)
        out.to_csv(outname, header=header, index=False)
    else:
        cnt = Counter()
        for i,dfp in enumerate(df):
            out,c = tok_df(dfp, text_cols, **kwargs)
            out.to_csv(outname, header=header if i==0 else None, index=False, mode='w' if i==0 else 'a')
            cnt.update(c)
    pickle.dump(cnt, open(outname.parent/'counter.pkl', 'wb'))

class SentencePieceTokenizer():#TODO: pass the special tokens symbol to sp
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, sp_model=None, vocab_sz=None, max_vocab_sz=30000,
                 model_type='unigram', char_coverage=None, cache_dir='tmp'):
        try: from sentencepiece import SentencePieceTrainer,SentencePieceProcessor
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
        self.sp_model,self.cache_dir = sp_model,Path(cache_dir)
        self.vocab_sz,self.max_vocab_sz,self.model_type = vocab_sz,max_vocab_sz,model_type
        self.char_coverage = ifnone(char_coverage, 0.99999 if lang in eu_langs else 0.9998)
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        if sp_model is None: self.tok = None
        else:
            self.tok = SentencePieceProcessor()
            self.tok.Load(str(sp_model))
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_vocab_sz(self, raw_text_path):
        cnt = Counter()
        with open(raw_text_path, 'r') as f:
            for line in f.readlines():
                cnt.update(line.split())
                if len(cnt)//4 > self.max_vocab_sz: return self.max_vocab_sz
        res = len(cnt)//4
        while res%8 != 0: res+=1
        return res

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
        from sentencepiece import SentencePieceTrainer
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        spec_tokens = ['\u2581'+s for s in self.special_toks]
        SentencePieceTrainer.Train(" ".join([
            f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}",
            f"--character_coverage={self.char_coverage} --model_type={self.model_type}",
            f"--unk_id={len(spec_tokens)} --pad_id=-1 --bos_id=-1 --eos_id=-1",
            f"--user_defined_symbols={','.join(spec_tokens)}"]))
        raw_text_path.unlink()
        return self.cache_dir/'spm.model'

    def setup(self, items, rules):
        if self.tok is not None: return {'sp_model': self.sp_model}
        raw_text_path = self.cache_dir/'texts.out'
        with open(raw_text_path, 'w') as f:
            for t in progress_bar(apply_rules(items, rules), total=len(items), leave=False):
                f.write(f'{t}\n')
        return {'sp_model': self.train(raw_text_path)}

    def pipe(self, items):
        for t in items: yield self.tok.EncodeAsPieces(t)
from transformers import T5TokenizerFast

tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PREFIX = "translate English to SQL: "
MAX_SRC, MAX_TGT = 128, 256


def load(path):
    with open(path) as f:
        return [l.strip() for l in f]


def summarize(texts, max_len=None, prefix=""):
    ids_list = []
    for t in texts:
        kwargs = dict(add_special_tokens=True)
        if max_len is not None:
            kwargs.update(max_length=max_len, truncation=True)
        ids_list.append(tok(prefix + t, **kwargs)["input_ids"])
    lengths = [len(x) for x in ids_list]
    vocab = {i for x in ids_list for i in x}
    return len(texts), sum(lengths) / len(lengths), len(vocab)


print("=== Table 1: before pre-processing ===")
for split in ["train", "dev"]:
    nls, sqls = load(f"data/{split}.nl"), load(f"data/{split}.sql")
    n, mean_nl, v_nl = summarize(nls)
    _, mean_sql, v_sql = summarize(sqls)
    print(f"{split}: n={n} mean_nl={mean_nl:.2f} mean_sql={mean_sql:.2f} vocab_nl={v_nl} vocab_sql={v_sql}")

print("=== Table 2: after pre-processing ===")
for split in ["train", "dev"]:
    nls, sqls = load(f"data/{split}.nl"), load(f"data/{split}.sql")
    _, mean_nl, v_nl = summarize(nls, max_len=MAX_SRC, prefix=PREFIX)
    _, mean_sql, v_sql = summarize(sqls, max_len=MAX_TGT)
    print(f"{split}: mean_nl={mean_nl:.2f} mean_sql={mean_sql:.2f} vocab_nl={v_nl} vocab_sql={v_sql}")

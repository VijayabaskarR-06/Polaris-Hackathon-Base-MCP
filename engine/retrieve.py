"""Lexical retrieval: BM25 over catalog text plus a product-name index, all numpy.

A query's score vector is built by scatter-adding precomputed per-posting BM25
weights, so scoring the whole catalog costs microseconds per query term.
"""
import math
from collections import Counter

import numpy as np

from engine.text import tokenize

# Field weights: a product's name and type describe it best.
DOC_FIELDS = (('name', 3), ('product_type', 2), ('description', 1), ('color', 1),
              ('pattern', 1), ('section', 1), ('category', 1), ('audience', 1))


class Postings:
    """Compressed-sparse-row postings: term -> (doc positions, weights)."""

    def __init__(self, n_docs, doc_terms, weigh):
        vocab, term_ids, doc_ids, values = {}, [], [], []
        for doc, terms in enumerate(doc_terms):
            for term, value in terms.items():
                term_ids.append(vocab.setdefault(term, len(vocab)))
                doc_ids.append(doc)
                values.append(value)
        term_ids = np.array(term_ids, dtype=np.int32)
        order = np.argsort(term_ids, kind='stable')     # docs stay ascending within a term
        self.docs = np.array(doc_ids, dtype=np.int32)[order]
        raw = np.array(values, dtype=np.float32)[order]
        counts = np.bincount(term_ids, minlength=len(vocab))
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.vocab = vocab
        self.df = {term: int(counts[i]) for term, i in vocab.items()}
        self.n = n_docs
        self.weights = weigh(self, raw)

    def span(self, term):
        i = self.vocab.get(term)
        if i is None:
            return None
        return slice(self.offsets[i], self.offsets[i + 1])

    def docs_of(self, term):
        s = self.span(term)
        return self.docs[s] if s is not None else np.empty(0, dtype=np.int32)


class BM25:
    def __init__(self, products, k1=1.2, b=0.75):
        self.n = n = len(products)
        token_cache = {}

        def tokens(text):
            got = token_cache.get(text)
            if got is None:
                got = token_cache[text] = tokenize(text)
            return got

        doc_terms, name_terms, lengths = [], [], np.zeros(n, dtype=np.float32)
        for doc, product in enumerate(products):
            counts = Counter()
            for field, weight in DOC_FIELDS:
                for term in tokens(str(product.get(field) or '')):
                    counts[term] += weight
            doc_terms.append(counts)
            lengths[doc] = sum(counts.values())
            name_terms.append(dict.fromkeys(tokens(product['name']), 1.0))
        avg = float(lengths.mean()) if n else 1.0

        def bm25_weights(post, tf):
            df = np.diff(post.offsets).astype(np.float32)
            idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
            term_idf = np.repeat(idf, np.diff(post.offsets))
            dl = lengths[post.docs]
            return term_idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))

        self.text = Postings(n, doc_terms, bm25_weights)
        self.names = Postings(n, name_terms, lambda post, raw: raw)
        self.idf = {t: math.log1p((n - df + 0.5) / (df + 0.5)) for t, df in self.text.df.items()}

    def known(self, term):
        return term in self.text.vocab

    def df(self, term):
        return self.text.df.get(term, 0)

    def scores(self, terms, weights=None):
        """BM25 score of every product for the given (stemmed) terms; weights scale individual terms."""
        out = np.zeros(self.n, dtype=np.float32)
        for term in dict.fromkeys(terms):
            s = self.text.span(term)
            if s is None:
                continue
            w = 1.0 if weights is None else weights.get(term, 1.0)
            out[self.text.docs[s]] += self.text.weights[s] * np.float32(w)
        return out

    def coverage(self, terms):
        """Number of distinct terms (out of terms) that each product's text contains."""
        out = np.zeros(self.n, dtype=np.int16)
        for term in dict.fromkeys(terms):
            out[self.text.docs_of(term)] += 1
        return out

    def name_hits(self, terms):
        """Number of distinct terms that appear in each product's name."""
        out = np.zeros(self.n, dtype=np.int16)
        for term in dict.fromkeys(terms):
            out[self.names.docs_of(term)] += 1
        return out

    def term_mask(self, term, names_only=False):
        mask = np.zeros(self.n, dtype=bool)
        mask[(self.names if names_only else self.text).docs_of(term)] = True
        return mask

    def any_mask(self, terms, names_only=False):
        mask = np.zeros(self.n, dtype=bool)
        index = self.names if names_only else self.text
        for term in dict.fromkeys(terms):
            mask[index.docs_of(term)] = True
        return mask

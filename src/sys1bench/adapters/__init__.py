"""Model adapters. Register new models with `@register("id")` or via the
`sys1bench.adapters` entry-point group so future models need no changes here."""

from . import (  # noqa: F401
    generic_http,
    hybrid_router,
    jev_openrouter,
    jev_typesafe,
    laya_local,
    majority_prior,
    mock,
    regex_keyword,
)
from .base import BaseAdapter, get_adapter, list_adapters, register  # noqa: F401

try:  # optional heavy deps
    from . import embed_knn, encoder_finetuned, llm_constrained, nli_zeroshot  # noqa: F401
except Exception:  # pragma: no cover
    pass

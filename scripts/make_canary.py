"""Build the fixed 200-item drift canary (100 tickets + 100 phishing emails, seed 9001). Regenerable; committed so
every session compares against the same items."""

from pathlib import Path

from sys1bench.generators import get_generator
from sys1bench.runners import write_manifest

items = get_generator("support_tickets", n=100, seed=9001).generate() + get_generator("phishing_email", n=100, seed=9001).generate()
out = Path(__file__).resolve().parents[1] / "data" / "canary" / "canary.jsonl"
write_manifest(items, out)
print(f"wrote {len(items)} canary items to {out}")

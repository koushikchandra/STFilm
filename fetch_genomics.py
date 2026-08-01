import re
import openreview
import pandas as pd

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Regex stems. (term_label, compiled_regex, is_generic)
TERMS = [
    ("genom",            r"genom",                         False),
    ("gene expression",  r"gene expression",               False),
    ("gene regulat",     r"gene regulat",                  False),
    ("transcriptom",     r"transcriptom",                  False),
    ("RNA-seq",          r"rna[\s\-]?seq",                 False),
    ("scRNA",            r"scrna",                         False),
    ("single-cell",      r"single[\s\-]cell",              False),
    ("DNA",              r"\bdna\b",                       True),
    ("epigenom",         r"epigenom",                      False),
    ("chromatin",        r"chromatin",                     False),
    ("GWAS",             r"\bgwas\b",                      False),
    ("variant calling",  r"variant calling",               False),
    ("mutation",         r"mutation",                      True),
    ("multi-omics",      r"multi[\s\-]?omics",             False),
    ("proteom",          r"proteom",                       False),
    ("sequencing",       r"sequencing",                    False),
]
COMPILED = [(label, re.compile(rx, re.IGNORECASE), generic) for label, rx, generic in TERMS]

def get_val(content, key):
    v = content.get(key, {})
    if isinstance(v, dict):
        return v.get('value', '')
    return v or ''

records = []
scanned = {}
for year in ['2025', '2026']:
    venue_id = f'ICLR.cc/{year}/Conference'
    notes = client.get_all_notes(content={'venueid': venue_id})
    scanned[year] = len(notes)
    for n in notes:
        c = n.content
        title = get_val(c, 'title')
        abstract = get_val(c, 'abstract')
        decision = get_val(c, 'venue')  # e.g. 'ICLR 2025 Poster'
        authors = get_val(c, 'authors')
        if isinstance(authors, list):
            authors = ', '.join(authors)
        haystack = f"{title}\n{abstract}"

        matched = []
        generic_only_flags = []
        for label, rx, generic in COMPILED:
            if rx.search(haystack):
                matched.append(label)
                generic_only_flags.append(generic)
        if not matched:
            continue

        n_distinct = len(set(matched))
        non_generic = [m for m, g in zip(matched, generic_only_flags) if not g]
        # high: 2+ distinct matches; low: only 1 generic term
        if n_distinct >= 2:
            confidence = 'high'
        elif len(non_generic) >= 1:
            confidence = 'high'  # single but non-generic specific term
        else:
            confidence = 'low'   # only a single generic term (DNA/mutation)

        url = f"https://openreview.net/forum?id={n.id}"
        records.append({
            'year': year,
            'title': title,
            'authors': authors,
            'decision': decision,
            'confidence': confidence,
            'matched_terms': '; '.join(sorted(set(matched))),
            'url': url,
            'abstract': abstract,
        })

df = pd.DataFrame(records)
out_csv = '/lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology/iclr_genomics_papers.csv'
df.drop(columns=['abstract']).to_csv(out_csv, index=False)

# Markdown grouped by year
md_lines = ["# ICLR Genomics-Related Accepted Papers\n"]
for year in ['2025', '2026']:
    sub = df[df.year == year]
    md_lines.append(f"\n## ICLR {year} ({len(sub)} matched papers)\n")
    for _, r in sub.iterrows():
        md_lines.append(f"- **{r['title']}** ({r['decision']}) [{r['confidence']}]")
        md_lines.append(f"  - Matched: {r['matched_terms']}")
        md_lines.append(f"  - {r['url']}")
out_md = '/lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology/iclr_genomics_papers.md'
with open(out_md, 'w') as f:
    f.write('\n'.join(md_lines))

# Report
print("==== REPORT ====")
for year in ['2025', '2026']:
    sub = df[df.year == year]
    low = sub[sub.confidence == 'low']
    print(f"ICLR {year}: scanned={scanned[year]}, matched={len(sub)}, low_confidence={len(low)}")
print(f"CSV: {out_csv}")
print(f"MD:  {out_md}")

# Save full df (with abstract) for downstream histo+omics filtering
df.to_pickle('/lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology/_genomics_full.pkl')

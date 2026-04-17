import re

def fix_title(t):
    match = re.search(r"^(.*?)(?:,\s+(The|A|An))(\s+\(\d{4}\))?$", t)
    if match:
        main, article, year = match.groups()
        year = year or ""
        return f"{article} {main}{year}"
    return t

tests = [
    "Silence of the Lambs, The (1991)",
    "Terminator, The (1984)",
    "Usual Suspects, The (1995)",
    "Die Hard 2 (1990)",
    "Englishman Who Went Up a Hill, But Came Down a Mountain, The (1995)",
    "Postino, Il (1994)",
    "Lion King, The (1994)",
    "Godfather, The (1972)",
    "Good, The Bad and The Ugly, The (1966)"
]

for t in tests:
    print(f"'{t}' -> '{fix_title(t)}'")

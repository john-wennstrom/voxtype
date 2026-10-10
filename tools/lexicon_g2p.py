"""Compile English terms into stress-free ARPAbet variants for voxtype."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import functools
import itertools
import os
import re
import subprocess
import tempfile
from pathlib import Path


PHONES = set("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())
LETTERS = "ay bee see dee ee eff gee aitch eye jay kay ell em en oh pee cue ar ess tee you vee double-you ex why zee".split()
LETTER_NAMES = dict(zip("abcdefghijklmnopqrstuvwxyz", LETTERS))
LETTER_NAMES["w"] = "double you"
SYMBOLS = {".": "dot", "_": "underscore", "-": "dash", "/": "slash", "#": "sharp", "+": "plus", "@": "at"}
DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
IPA = {
    "tʃ": "CH", "dʒ": "JH", "eɪ": "EY", "aɪ": "AY", "ɔɪ": "OY", "aʊ": "AW", "oʊ": "OW", "əʊ": "OW",
    "ɑ": "AA", "ɒ": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɔ": "AO", "ɛ": "EH", "ɜ": "ER", "ɚ": "ER",
    "ɝ": "ER", "ɪ": "IH", "i": "IY", "ʊ": "UH", "u": "UW", "θ": "TH", "ð": "DH", "ʃ": "SH", "ʒ": "ZH",
    "ŋ": "NG", "ɹ": "R", "j": "Y", **{phone.lower(): phone for phone in "B D F G K L M N P R S T V W Z".split()}, "h": "HH",
}


def normalize(phones):
    result = tuple(re.sub(r"\d", "", phone).upper() for phone in phones)
    if not result or any(phone not in PHONES for phone in result):
        raise ValueError(f"Invalid ARPAbet pronunciation: {phones}")
    return result


def ipa_to_arpabet(text):
    result = []
    text = re.sub(r"[\sˈˌːˑ_()\[\]]", "", text)
    while text:
        match = next((symbol for symbol in sorted(IPA, key=len, reverse=True) if text.startswith(symbol)), None)
        if match is None:
            raise ValueError(f"Unsupported espeak IPA sequence: {text}")
        result.append(IPA[match])
        text = text[len(match):]
    return normalize(result)


def read_terms(path):
    seen = set()
    with path.open(encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            canonical = row[0].strip()
            folded = canonical.casefold().lstrip("!")
            if folded in seen:
                raise ValueError(f"{path}:{line_number}: duplicate term {canonical}")
            seen.add(folded)
            weight = float(row[2]) if len(row) > 2 and row[2].strip() else 1.0
            if not 0 < weight <= 1:
                raise ValueError(f"{path}:{line_number}: weight must be in (0, 1]")
            forms = [form.strip() for form in row[1].split("|") if form.strip()] if len(row) > 1 else []
            yield canonical, forms, weight


def split_identifier(term, symbols=True):
    term = re.sub(r"([a-z])([A-Z])", r"\1 \2", term)
    term = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", term)
    parts = []
    for part in re.findall(r"[A-Za-z]+|\d|[^\w\s]|_", term):
        part = part.strip()
        if part in DIGITS:
            parts.append(DIGITS[part])
        elif part in SYMBOLS:
            if symbols:
                parts.extend(SYMBOLS[part].split())
        elif part:
            parts.append(part)
    return parts


class Pronouncer:
    def __init__(self, dictionary, backend="g2p_en"):
        self.dictionary = dictionary
        self.backend = backend
        self.neural = None

    @functools.lru_cache(maxsize=None)
    def word(self, word):
        known = self.dictionary.get(word.lower())
        if known:
            return tuple(dict.fromkeys(normalize(phones) for phones in known))
        variants = []
        if self.backend in ("g2p_en", "all"):
            if self.neural is None:
                from g2p_en import G2p
                self.neural = G2p()
            variants.append(normalize(phone for phone in self.neural(word) if phone.strip()))
        if self.backend in ("espeak", "all"):
            output = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", word], check=True, capture_output=True, text=True)
            variants.append(ipa_to_arpabet(output.stdout))
        if not variants:
            raise ValueError(f"No pronunciation for {word!r}; choose a G2P backend or add a spoken form")
        return tuple(dict.fromkeys(variants))

    def phrase(self, words, limit):
        pronunciations = [self.word(word) for word in words]
        for combination in itertools.islice(itertools.product(*pronunciations), limit):
            yield tuple(itertools.chain.from_iterable(combination))


def variants(term, forms, pronouncer, cap=8):
    result = []
    seen = set()

    def add(source, phones, required=False):
        if phones not in seen and (required or len(result) < cap):
            result.append((source, phones))
            seen.add(phones)

    for form in forms:
        for phones in pronouncer.phrase(form.split(), cap):
            add("user", phones, required=True)
    for phones in pronouncer.dictionary.get(term.lower(), []):
        add("cmudict", normalize(phones))
    for with_symbols in (True, False):
        parts = split_identifier(term, with_symbols)
        if parts:
            try:
                for phones in pronouncer.phrase(parts, cap):
                    add("split", phones)
            except (ValueError, ImportError):
                pass
    names = []
    spelled = False
    for part in split_identifier(term, False):
        if part.isalpha() and ((part.isupper() and 2 <= len(part) <= 5) or not re.search(r"[aeiou]", part, re.I)):
            names.extend(" ".join(LETTER_NAMES[letter.lower()] for letter in part).split())
            spelled = True
        else:
            names.append(part)
    if spelled:
        try:
            for phones in pronouncer.phrase(names, cap):
                add("acronym", phones)
        except (ValueError, ImportError):
            pass
    if not result:
        raise ValueError(f"No pronunciation variants for {term!r}")
    return result


def initialize_worker(dictionary, backend, cap):
    global worker_pronouncer, worker_cap
    worker_pronouncer = Pronouncer(dictionary, backend)
    worker_cap = cap


def compile_term(task):
    term_id, (term, forms, weight) = task
    if term.startswith("!"):
        return []
    return [(term_id, term, source, " ".join(phones), weight)
            for source, phones in variants(term, forms, worker_pronouncer, worker_cap)]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            csv.writer(output, delimiter="\t", lineterminator="\n").writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terms", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("g2p_en", "espeak", "all", "none"), default="g2p_en")
    parser.add_argument("--max-variants", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--export-query-dict", type=Path)
    args = parser.parse_args()
    if args.max_variants < 1:
        parser.error("--max-variants must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        import cmudict
        dictionary = cmudict.dict()
        tasks = list(enumerate(read_terms(args.terms)))
        if args.workers == 1:
            initialize_worker(dictionary, args.backend, args.max_variants)
            write_rows(args.output, itertools.chain.from_iterable(map(compile_term, tasks)))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                     initargs=(dictionary, args.backend, args.max_variants)) as pool:
                write_rows(args.output, itertools.chain.from_iterable(pool.map(compile_term, tasks, chunksize=64)))
        if args.export_query_dict:
            write_rows(args.export_query_dict, ((word, " ".join(normalize(pronunciations[0])))
                       for word, pronunciations in sorted(dictionary.items())
                       if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word)))
    except (ImportError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"lexicon G2P: {error}\n")


if __name__ == "__main__":
    main()
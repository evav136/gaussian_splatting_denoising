"""
organize.py
Uredi training_set/ v strukturo noisy/ clean/ depth/ z zaporednim poimenovanjem.

Uporaba:
  python organize.py --src ../training_set --dst ../training_data
"""

import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def parse_filename(name):
    """
    Iz imena datoteke izvleče (tip, indeks, variant).
    """

    m = re.match(r'^(noisy|clean|depth)_(\d+)(?:\s*\((\d+)\))?\.png$', name)

    if not m:
        return None
    
    tip = m.group(1)
    idx = m.group(2)
    variant = m.group(3) or '0'

    return tip, idx, variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='../training_set', help='Vhodna mapa z vsemi PNG datotekami')
    parser.add_argument('--dst', default='../training_data', help='Izhodna organizirana mapa')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # Grupiraj po (indeks, variant) -> {'noisy': path, 'clean': path, 'depth': path}
    groups = defaultdict(dict)
    skipped = []

    for f in sorted(src.glob('*.png')):
        parsed = parse_filename(f.name)

        if parsed is None:
            skipped.append(f.name)
            continue

        tip, idx, variant = parsed
        key = (idx, variant)
        groups[key][tip] = f

    # Poišči veljavne pare (vse tri komponente prisotne)
    valid = []
    incomplete = []

    for key, parts in sorted(groups.items()):
        if 'noisy' in parts and 'clean' in parts and 'depth' in parts:
            valid.append(parts)
        else:
            incomplete.append((key, list(parts.keys())))

    print(f"Najdeno skupin:  {len(groups)}")
    print(f"Veljavnih parov: {len(valid)}")
    print(f"Nepopolnih:      {len(incomplete)}")

    if incomplete:
        print("  Nepopolni pari:")
        for key, has in incomplete[:5]:
            print(f"    indeks={key[0]} variant={key[1]}: ima {has}")
    if skipped:
        print(f"Preskočenih:       {len(skipped)}")

    # Ustvari izhodne mape
    for sub in ['noisy', 'clean', 'depth']:
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # Kopiraj in preimenuj zaporedno
    print(f"\nKopiram v {dst} ...")
    for i, parts in enumerate(valid):
        new_name = f"{i:04d}.png"
        shutil.copy2(parts['noisy'], dst / 'noisy' / new_name)
        shutil.copy2(parts['clean'], dst / 'clean' / new_name)
        shutil.copy2(parts['depth'], dst / 'depth' / new_name)

    print(f"Organiziranih {len(valid)} parov:")
    print(f"  {dst}/noisy/0000.png ... {(len(valid)-1):04d}.png")
    print(f"  {dst}/clean/0000.png ... {(len(valid)-1):04d}.png")
    print(f"  {dst}/depth/0000.png ... {(len(valid)-1):04d}.png")


if __name__ == '__main__':
    main()

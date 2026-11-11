"""Uzak ZIP'ten, adinda verilen alt dizgeyi iceren dosyalardan rastgele N tanesini cikarir (Range istekleriyle).
Kullanim: python -I uzak_zip_ornek.py <url> <alt_dizge> <N> <hedef_klasor> [tohum]
"""
import io
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uzak_zip_listele import UzakDosya  # noqa: E402


def main():
    url, alt, n, hedef = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    tohum = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    z = zipfile.ZipFile(io.BufferedReader(UzakDosya(url), buffer_size=1 << 20))
    adaylar = [i for i in z.infolist() if alt in i.filename and not i.is_dir()]
    random.Random(tohum).shuffle(adaylar)
    os.makedirs(hedef, exist_ok=True)
    for i in adaylar[:n]:
        ad = os.path.basename(i.filename).replace(" ", "_")
        with open(os.path.join(hedef, ad), "wb") as f:
            f.write(z.read(i))
        print(os.path.join(hedef, ad))


if __name__ == "__main__":
    main()

"""Uzak bir ZIP arsivinin icerigini (yalnizca merkez dizin) HTTP Range istekleriyle listeler.
Arsivin tamamini indirmeden klasor/sinif yapisini gormek icin kullanilir.
Kullanim: python -I uzak_zip_listele.py <url> [--klasor-ozeti]
"""
import io
import sys
import urllib.request
import zipfile
from collections import Counter

UA = {"User-Agent": "research-dataset-client/1.0"}


class UzakDosya(io.RawIOBase):
    def __init__(self, url):
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        r = urllib.request.urlopen(req, timeout=120)
        self.url = r.geturl()  # yonlendirme sonrasi gercek adres
        cr = r.headers.get("Content-Range")
        if not cr:
            raise SystemExit("Sunucu Range desteklemiyor")
        self.boyut = int(cr.split("/")[1])
        self.konum = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, ofs, whence=0):
        if whence == 0:
            self.konum = ofs
        elif whence == 1:
            self.konum += ofs
        else:
            self.konum = self.boyut + ofs
        return self.konum

    def tell(self):
        return self.konum

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.boyut - self.konum
        if n == 0 or self.konum >= self.boyut:
            return b""
        son = min(self.konum + n, self.boyut) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.konum}-{son}"})
        veri = urllib.request.urlopen(req, timeout=120).read()
        self.konum += len(veri)
        return veri

    def readinto(self, b):
        veri = self.read(len(b))
        b[: len(veri)] = veri
        return len(veri)


def main():
    url = sys.argv[1]
    ozet = "--klasor-ozeti" in sys.argv
    f = io.BufferedReader(UzakDosya(url), buffer_size=1 << 20)
    z = zipfile.ZipFile(f)
    adlar = [i for i in z.infolist() if not i.is_dir()]
    print("toplam dosya:", len(adlar))
    if ozet:
        c = Counter("/".join(i.filename.split("/")[:-1]) for i in adlar)
        for k, v in sorted(c.items()):
            print(f"{v:6d}  {k}")
        uz = Counter(i.filename.rsplit(".", 1)[-1].lower() for i in adlar)
        print("uzantilar:", dict(uz))
    else:
        for i in adlar:
            print(i.file_size, i.filename)


if __name__ == "__main__":
    main()

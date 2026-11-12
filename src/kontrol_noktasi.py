# -*- coding: utf-8 -*-
"""En iyi model (kontrol noktası) ve erken durdurma takibi (04 ve 07 betikleri ortak kullanır).

Kural:
  * En iyi model = en düşük doğrulama kaybı (kesin minimum): kayip < en_iyi ise kaydedilir.
  * Erken durdurma sabrı ayrı tutulur: yalnız referanstan 1e-4'ten fazla iyileşme sabrı sıfırlar.
    Böylece çok küçük iyileşmeler en iyi modeli günceller ama eğitimi uzatmaz.
"""


class EnIyiTakip:
    def __init__(self, tolerans: float = 1e-4):
        self.tolerans = tolerans
        self.en_iyi = float("inf")
        self.en_iyi_epoch = 0
        self.sabir = 0
        self._sabir_ref = float("inf")  # sabrı sıfırlayan son "anlamlı" iyileşmenin kaybı

    def guncelle(self, kayip: float, epoch: int) -> bool:
        """Epoch sonucunu işler; kontrol noktası kaydedilmeliyse True döndürür."""
        iyi = kayip < self.en_iyi
        if iyi:
            self.en_iyi, self.en_iyi_epoch = kayip, epoch
        if kayip < self._sabir_ref - self.tolerans:
            self._sabir_ref, self.sabir = kayip, 0
        else:
            self.sabir += 1
        return iyi

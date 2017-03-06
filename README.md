# Első feladat

Készítsen hegyi kerékpárpálya tervezőt, amely felülnézetben, merőleges vetülettel mutatja a pályát és környékét! A terepet explicit magasságmezővel adjuk meg. A kontrolpontok xy vetületei szabályos N x N-es (N > 3) 2D rácsot alkotnak az 1km x 1km tartományban. A tervezőprogramban a magasságot térképszerű, de megválasztható színkódolással kell bemutatni (például de nem kötelezően: alacsony zöld, magasabb világosbarna, magas sötétbarna). A színkódoláshoz a magasságot 50 méterenként kell kiértékelni, a mintapontok között a szín lineárisan változik.

A tervező a bal egérgomb lenyomásokkal a terepre vetíti a pálya kontrolpontjait, amelyet a program fehér zárt Lagrange interpolációs görbével köt össze és printf-fel kiírja az aktuális pálya hosszát. SPACE lenyomására egy alkalmas színű nyílszerű konkáv poligonnal ábrázolt virtuális biciklista indul el a pályán a nyilat mindig a haladási irányba állítva. A haladási irány 3D-ben analitikusan (nem pedig közelítő differenciahányadossal) számítandó. A biciklista a két kontrolpont között pontosan annyi időt tölt el, amennyi a két gomblenyomás között eltelt. A pálya meredekségét a képernyőn tetszőleges helyen elhelyezett derékszögű háromszöggel szemléltesse, amelynek az emelkedési szög az egyik szöge, és állása mutatja, hogy emelkedővel küzd-e a kerékpáros vagy lejtőn gurul lefelé.

Beadási határidő: **2017. 03. 26. 23:59**

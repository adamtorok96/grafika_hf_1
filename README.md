# Első feladat

Készítsen hegyi kerékpárpálya tervezőt, amely felülnézetben, merőleges vetülettel mutatja a pályát és környékét!
A terepet Bézier felülettel adjuk meg.
A kontrollpontok xy vetületei szabályos N x N-es (N > 3) 2D rácsot alkotnak az 1km x 1km tartományban.
A tervezőprogramban a magasságot térképszerű, de megválasztható színkódolással kell bemutatni (például de nem kötelezően: alacsony zöld, magasabb világosbarna, magas sötétbarna).
A színkódoláshoz a magasságot 50 méterenként kell kiértékelni, a mintapontok között a szín lineárisan változik.

A tervező a bal egérgomb lenyomásokkal a terepre vetíti a pálya kontrollpontjait, amelyet a program fehér Lagrange interpolációs görbével köt össze és printf-fel kiírja az aktuális pálya hosszát.
SPACE lenyomására egy alkalmas színű nyílszerű konkáv poligonnal ábrázolt virtuális biciklista indul el a pályán a nyilat mindig a haladási irányba állítva.
A haladási irány 3D-ben analitikusan (nem pedig közelítő differenciahányadossal) számítandó.
A biciklista a két kontrollpont között pontosan annyi időt tölt el, amennyi a két gomblenyomás között eltelt.
A pálya meredekségét a képernyőn tetszőleges helyen elhelyezett derékszögű háromszöggel szemléltesse, amelynek az emelkedési szög az egyik szöge, és állása mutatja, hogy emelkedővel küzd-e a kerékpáros vagy lejtőn gurul lefelé.

Beadási határidő: **2017. 03. 26. 23:59**

# A megoldás módja

A feladatot C++ nyelven kell megvalósítani OpenGL és GLUT felhasználásával az alábbi sablon módosításával. A feladat megoldása során implementálni kell az onInitialization(), onDisplay(), onKeyboard(), onMouse(), onMouseMotion() és onIdle() függvényeket. Amennyiben a feladat megköveteli természetesen létrehozhatsz segédfüggvényeket is. Fontos azonban, hogy csak a jelzett részen belül található programrészeket értékeljük.

####  A forráskód feltöltése

Az elkészült forráskód feltöltésére a Kódfeltöltés menüpont alatt van lehetőséged. A házi feladat határidejének lejártáig tetszőleges alkalommal tölthetsz fel megoldás, értékelni az utolsó változatot fogjuk.

#### A fordítási és futási eredmények ellenőrzése

A fordítás sikerességét az Eredmények menüpont alatt tudod ellenőrizni.

# Néhány tanács

A programokat GNU g++ 4.7.2 verziószámú fordítóprogrammal fordítjuk, mely szigorúbban követi a C++ szabványt mint a Visual Studio különböző változatai, ezért előfordulhat, hogy sikertelen lesz a fordítás annak ellenére, hogy a Visual Studio fordítójával lefordul. A pontos hibáról az Eredmények menüpont alatt tájékozódhatsz.

A feladatokat ISO8859-1 vagy ISO8859-2 formátumban várjuk, az UTF-16 és UNICODE formátumú programok hibás fordítást okoznak.

A sablonban szereplő elválasztó sorokat (//~~~~~~...) célszerű a beküldött programban is megőrizni. Ellenkező esetben előfeldolgozási hiba lehet, bár a gyakoribb esetekre a parsert felkészítettük.

#### Gyakori hibák

* Gyakori hiba konstans argumentumot átadni referenciát váró függvénynek. Pl. void fv(int& a) függvényt void fv(2)-vel hívni. Ez nem okoz hibát Visual Studio alatt az alap beállításokkal, de a szabvány szerint hibás.
* A tipikus C++ hibákról jó összefoglalót találhatsz ezen az oldalon.
* Az OpenGL specifikáció nem rendelkezik a ModelView és Projection mátrixok alapértelmezett értékeiről. Ezek a program indulásakor tetszőlegesek lehetnek. Több beküldött feladatnál tapasztaltuk, hogy hibásan feltételezte az egység mátrixot. A kiadott forráskód-sablonban ezért inicializáljuk a mátrixokat a futás kezdetén.
* Több beküldött megoldásban tapasztaltuk az előfordítói #define direktívával definiált makrók hibás használatát. Semmi sem garantálja, hogy más rendszereken vagy akár csak más GLUT verzióban a numerikus konstansok megegyeznek, ezért hibás programozói gyakorlat ezen numerikus konstansok használata. Helyettük az előfordítói makrókat kell használni.
* Az onIdle() függvény akkor hívódik amikor semilyen esemény nem érkezik az rendszertől. Ennek megfelelően semmi sem garantálja, hogy mikor fog lefutni. Ebből következően itt olyan műveleteket végezni, melyek nélkül a renderelés hibás lesz (pl. a mátrixok beállítása) nem érdemes.
* Nehány hasznos tanács a GLUT használatához.
* Csak a sablonban regisztralt callback fuggvenyeket erdemes hasznalni, mivel a teszt rendszerben a tobbi glut fuggveny meghivasa sikertelen lesz.
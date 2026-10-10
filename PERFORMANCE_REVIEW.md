# Analiza wydajności Minerust

Analiza kodu i pomiar CPU: 2026-10-10. Przejrzane ścieżki obejmują konfigurację
silnika, generowanie i przechowywanie świata, kolejki workerów, przygotowanie
snapshotów i meshów, aktualizację klatki, cache widocznych chunków, bufory
indirect oraz zapis/wczytywanie świata.

## Konfiguracja i skala pracy

Przy obecnych stałych `CHUNK_SIZE = 16`, `WORLD_HEIGHT = 256` i
`SUBCHUNK_HEIGHT = 16` kolumna zawiera 16 subchunków oraz 65 536 pozycji bloków.
Przechowywanie `Uniform` ogranicza pamięć zajmowaną przez jednolite subchunki;
`Dense` jest materializowane dopiero przy zmianie pojedynczych bloków.

Kwadratowy zasięg renderowania 32 obejmuje maksymalnie 4225 kolumn. Zasięg
generowania 34 obejmuje 4761 kolumn, a kwadrat o promieniu usuwania 37 — 5625.
To uzasadnia ograniczanie pracy głównego wątku oraz unikanie pełnych skanów.
Zasięgi, geometria świata, parametry szumu, jakość obrazu i tryb prezentacji
pozostały zgodne z konfiguracją użytkownika.

## Wprowadzone optymalizacje

| Ścieżka | Poprzedni koszt / problem | Zmiana |
|---|---|---|
| Snapshot subchunka | Odczyt mapy chunków dla każdej pozycji w buforze 18³ i ponownie dla cache nieba | Jednorazowe pobranie 9 sąsiednich kolumn, ponowne używanie referencji |
| Widoczność nieba przy kamerze | Do 5 skanów pionowych, każdy z odczytami mapy | 5 odczytów istniejącego cache najwyższego nieprzezroczystego bloku |
| Usuwanie chunków | Osobny pełny skan na potrzeby statystyk, nawet gdy właściwe usuwanie pomijało nieruchomego gracza | Jedna operacja `retain`; statystyki liczone tylko dla usuwanych kolumn |
| Wybór zleceń generowania | Sortowanie wszystkich brakujących kolumn przed wyborem małego podzbioru | Partycjonowanie w czasie liniowym i sortowanie wybranego podzbioru |
| Budżet zleceń | Stała deklarowała 8, kod zgłaszał do 16 żądań na klatkę | Kod przestrzega `MAX_CHUNKS_PER_FRAME` i wolnych miejsc w kolejce |
| Pule wątków | Minimum po 2 wątki na pulę; po dołączeniu do serwera mesher dostawał wszystkie dostępne wątki CPU | Wspólny budżet CPU i ten sam dobór workerów we wszystkich zmienianych ścieżkach |
| Zaległe meshe | Dwa kanały po 256 elementów nie ograniczały całej liczby oczekujących zadań do 256 | Limit `pending` obejmuje także wyniki jeszcze nieodebrane przez główny wątek |
| Puste meshe | Alokacje dwóch wektorów przed sprawdzeniem pustego snapshotu | Sprawdzenie pustego snapshotu przed alokacją |

`constants.rs` udostępnia teraz limity zaległych zleceń i obu pul workerów.
Przykładowo CPU z 4 dostępnymi wątkami uruchamia 1 worker generowania i 1
meshowania; przy 8 są to 3 i 3. Limity maksymalne to odpowiednio 8 i 6.
Na maszynach z najwyżej 3 wątkami każda pula nadal potrzebuje jednego workera;
rezerwacja dwóch wątków dla pracy pierwszoplanowej jest wtedy niemożliwa.
Mniejsze pule i twardszy limit kolejki mogą wydłużyć zapełnianie świata na słabym
CPU, ograniczając jednocześnie konkurencję o CPU i ilość zaległej pracy.

Zachowana została publiczna metoda `update_chunks_around_player`. Nowa
`unload_chunks_around_player` zwraca dodatkowo statystyki geometrii usuniętej
w tej samej operacji.

## Pomiar i weryfikacja

Porównawczy benchmark wykonuje po 20 000 snapshotów z poprzednią i nową
implementacją. Fixture obejmuje różne rodzaje bloków, ujemne współrzędne,
sąsiadów i brakującą kolumnę. Kod pakietu Minerust był kompilowany z
`opt-level = 3`; zależności zachowały profil testowy.

W jednym pomiarze poprzednia implementacja zajęła **724,99 ms**, nowa
**213,42 ms**, czyli około **3,40× mniej czasu** na przygotowanie snapshotów.
Wynik dotyczy tego mikrobenchmarku CPU. Nie zmierzono FPS ani czasu GPU;
wydajność całej gry zależy też od renderowania, szumu, uploadów i sceny.

Odtworzenie pomiaru:

```bash
cargo --config 'profile.test.package.minerust.opt-level=3' test --offline --lib benchmark_mesh_snapshot -- --ignored --nocapture
```

Sprawdzenia:

- `cargo test --offline --all-targets`: 35 testów przeszło, 1 ręczny benchmark
  jest domyślnie pomijany.
- Test porównawczy snapshotów sprawdza cały cache bloków, wysokości nieba,
  rewizję mesha i wykrywanie pustych subchunków dla wszystkich poziomów Y,
  również na granicach świata i przy brakujących sąsiadach.
- Test widoczności nieba porównuje wynik ze skanem bloków, także po usunięciu
  najwyższego nieprzezroczystego bloku.
- Testy obejmują usuwanie geometrii, granicę promienia, ponowne wywołanie bez
  ruchu, budżet workerów, przeciążenie kolejki i wybór najbliższych żądań.
- `cargo clippy --offline --all-targets` kończy się poprawnie. Projekt nadal
  zgłasza ostrzeżenia dotyczące m.in. nieużywanego kodu, przestarzałych API
  oraz stylu istniejących fragmentów.
- Zmodyfikowane pliki Rust poza istniejącymi zmianami użytkownika w
  `init.rs` i `input.rs` sprawdzono przez `rustfmt`.

## Pozostałe miejsca do profilowania

1. **Wyszukiwanie brakujących chunków w `app/update.rs`.** Gdy liczba
   oczekujących żądań spada poniżej 32, przeszukiwany jest cały obszar
   generowania. Dotyczy to także w pełni załadowanego świata. Kolejnym krokiem
   może być trwała kolejka kandydatów, unieważniana przy ruchu, zmianie świata
   oraz asynchronicznych wstawieniach.
2. **Przebudowa cache widocznych kolumn w `app/render.rs`.** Wstawienia chunków
   oznaczają ponowne sprawdzenie obszaru i sortowanie kolumn. Przy szybkim
   streamingu warto zmierzyć udział tej operacji w czasie klatki i rozważyć
   aktualizację przyrostową.
3. **Edycja bloków w `rebuild_player_dirty_meshes_now`.** Meshe dotkniętych
   subchunków są budowane synchronicznie dla natychmiastowego efektu edycji.
   Zmiana na worker wymaga zachowania responsywności interakcji oraz ochrony
   przed starymi wynikami.
4. **F9 w `app/game.rs`.** `World::new_with_seed` generuje pełny zasięg
   renderowania synchronicznie na wątku obsługi zdarzeń. Docelowo wczytywanie
   powinno korzystać ze streamingu i odtwarzać zapisane edycje przed meshowaniem.
5. **Asynchroniczny pierścień w `world/terrain.rs`.** Osobny wątek generuje
   kolumnę przed sprawdzeniem, czy już istnieje w świecie. Może wykonywać pracę
   równolegle z `ChunkLoader`; warto ujednolicić planowanie i obsługę wyników.
6. **GPU.** Istnieją timestampy i HUD wydajności. Dobór MSAA, Hi-Z, rozmiarów
   aren oraz budżetów uploadu powinien bazować na pomiarach w działającej grze
   na docelowym GPU. W tej analizie nie zmieniano ich na podstawie domysłów.


## Uzupełniający przegląd i optymalizacje

Poniższe zmiany wykonano po zastaniu optymalizacji opisanych wyżej.
Zachowano istniejące zmiany robocze. Ponownie uruchomiono testy oraz oba
mikrobenchmarki CPU. Nie zmieniano zasięgów świata, jakości obrazu ani
parametrów proceduralnego terenu.

### Planowanie generowania

`ChunkLoader::request_missing_chunks` korzysta z jednego, współdzielonego
porządku przesunięć względem gracza, posortowanego według `(odległość², x, z)`.
Kursor przechodzi po nim w kolejnych klatkach, zgłaszając najwyżej
`MAX_CHUNKS_PER_FRAME` zleceń i respektując `MAX_PENDING_CHUNKS`.
Załadowane i oczekujące kolumny są pomijane. Wyniki odebrane w bieżącej klatce,
ale jeszcze niewstawione do świata, też są traktowane jako zajęte pozycje.

Każda pozycja jest sprawdzana raz na pobyt gracza w danym chunku. Po wyczerpaniu
skanu kolejne klatki nie odczytują mapy świata na potrzeby generowania.
Zmiana chunka gracza resetuje kursor, a zastąpienie świata tworzy nowy loader.
Brak budżetu nie przesuwa kursora. Usuwanie chunków w aplikacji dotyczy
obszaru poza promieniem generowania; zewnętrzne usunięcie wewnątrz tego
obszaru wymaga resetu skanu przez `clear_pending` lub zastąpienia loadera.

Złożoność dla nieruchomego gracza spada z ponawianego skanu O(r²) na klatkę
do łącznego O(r²) na przejście całego obszaru i O(1) po jego wyczerpaniu.
Przy częstym przekraczaniu granic chunków skan rozpoczyna się od nowa.
Zlecenia już znajdujące się w kolejce zachowują wcześniejsze priorytety;
nie wprowadzano ich anulowania ani zmiany kolejności.

### Widoczność i cykl życia workerów

- Cache widocznych kolumn iteruje po jednorazowo posortowanych przesunięciach.
  Przebudowa nadal sprawdza O(r²) pozycji, ale nie sortuje ponownie do 4225
  kolumn. Zachowuje identyczną kolejność również dla równych odległości.
- Tworzenie świata uruchamia wyłącznie zarządzaną pulę `ChunkLoader` po
  przygotowaniu obszaru startowego. Usunięto równoległe wywołanie generatora
  pierścienia, które dublowało generowanie, omijało budżet commitów i mogło
  wstawiać kolumny po zastąpieniu świata. Publiczną metodę pierścienia
  pozostawiono dla zgodności; aplikacja jej już nie wywołuje.
- F9 odtwarza generatory chunków i meshów z seedem wczytanego świata,
  odrzuca wyniki poprzednich loaderów i resetuje statystyki oraz cache
  widoczności. Poprzednio loadery pozostawały związane z poprzednim seedem.
- `constants.rs` sprawdza w czasie kompilacji dodatnie wymiary, podzielność
  wysokości przez subchunki i rozmiary siatki jaskiń oraz kolejność promieni
  renderowania, generowania i usuwania.

### Aktualna weryfikacja

`cargo test --offline --all-targets`: **40 testów przeszło**, dwa ręczne
benchmarki są pomijane w zwykłym uruchomieniu. Nowe testy sprawdzają kilka
kolejnych partii zleceń względem pełnego sortowania, ujemne współrzędne,
wcześniej oczekujące zadania, odebrane wyniki, ruch gracza, pełną kolejkę,
zerowy budżet, reset skanu i wyczerpanie załadowanego obszaru. Test renderera
porównuje pełną kolejność kolumn dla świata pełnego, częściowego i pustego.

`cargo clippy --offline --all-targets` zakończył się poprawnie; pozostają
ostrzeżenia istniejącego kodu. `git diff --check` nie wykazał błędów.

Nowy mikrobenchmark planowania dla **20 000 klatek załadowanej okolicy**:
pełne skanowanie **1659,23 ms**, planowanie przyrostowe **0,167 ms** łącznie.
Fixture wykorzystuje zbiór współrzędnych; pomiar obejmuje pierwszy skan oraz
kolejne wywołania po jego wyczerpaniu, bez workerów generujących teren.
Spadek czasu wynika z usunięcia powtarzanej pracy. Nie jest to pomiar FPS,
GPU ani scenariusza ciągłego przemieszczania gracza.

Ponowiony benchmark snapshotów już zastanej implementacji: poprzednia
implementacja **705,74 ms**, zoptymalizowana **208,91 ms**, około **3,38×**.

```bash
cargo --config 'profile.test.package.minerust.opt-level=3' test --offline --lib benchmark_loaded_generation_scan -- --ignored --nocapture
cargo --config 'profile.test.package.minerust.opt-level=3' test --offline --lib benchmark_mesh_snapshot -- --ignored --nocapture
```

Wcześniejsze punkty 1, 2 i 5 listy miejsc do profilowania zostały odpowiednio
rozwiązane, częściowo zoptymalizowane i usunięte ze ścieżki aplikacji.
Pozostają synchroniczne meshowanie edycji, pełna generacja podczas F9 oraz
profilowanie GPU na docelowym sprzęcie. F9 nadal generuje kwadrat wokół
początku świata i nakłada zapisane edycje tylko na istniejące kolumny;
wczytywanie dalekich edycji wymaga osobnego rozwiązania odtwarzania danych.
W zapisie pomijane są puste subchunki, a usuwanie kolumn nie zachowuje osobno
edycji gracza — to istniejące problemy trwałości świata, wymagające zmiany
modelu zapisu, a nie doboru stałych wydajnościowych.

Nie wykonano interaktywnego testu gry ani pomiaru czasów klatek/GPU.

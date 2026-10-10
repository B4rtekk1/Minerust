# Przegląd i optymalizacje GPU Minerust

Data: 2026-10-10. Przegląd obejmuje graph renderowania, inicjalizację i resize
tekstur, shader resolve głębi MSAA, Hi-Z i culling indirect, aktywny shader
wody, terrain, sky, composite oraz timestampy GPU. Zachowano wcześniejsze
zmiany robocze i optymalizacje CPU.

## Główne znaleziska

1. **Resolve generował dane bez odbiorcy.** `depth_resolve.wgsl` odczytywał
   cztery próbki MSAA dla maksimum i ponownie dla minimum. Maksimum trafiało
   do Hi-Z, a minimum do pełnowymiarowej tekstury `ssr_depth`. Aktywny
   `water.wgsl` deklaruje i odczytuje tylko uniformy, opaque scene color oraz
   sampler; nie deklaruje bindingu 9 z głębią. Composite również nie odczytuje
   tego bufora. Nazwy i komentarze sugerowały pełne SSR, lecz aktualna woda
   używa zniekształconego koloru sceny i proceduralnego odbicia nieba.
2. **Pierwszy etap Hi-Z korzystał z kosztownego pośrednika.** Pełnowymiarowy
   seed R32Float był zapisywany przez resolve, a następnie odczytywany w
   osobnym passie redukcji do pierwszego mniejszego mipa.
3. **F4 nie usuwał kosztu generowania.** Wyłączenie occlusion cullingu
   wyłączało test okluzji, ale resolve i cała piramida nadal pracowały.
4. **Świeżość piramidy nie była jawna.** Pierwsza klatka po resize lub zmianie
   świata mogła używać nieaktualnego lub jeszcze niewygenerowanego Hi-Z.

## Wprowadzone zmiany

### Resolve połączony z pierwszą redukcją

`depth_resolve.wgsl` oblicza od razu maksimum próbek MSAA dla obszaru
pokrywającego texel nowej podstawy Hi-Z. Dla parzystych wymiarów jest to
2×2 piksele × 4 próbki. Podstawa ma `max(1, floor(surface / 2))` texeli na
każdej osi i odpowiada dotychczasowemu mipowi 1. Kolejne mipy są generowane
przez istniejący `hiz.wgsl`.

Nieparzyste rozmiary zachowują dotychczasowe nakładające się obszary 3×2,
2×3 lub 3×3 oraz clamping do granic źródła. Pominięcie skrajnych próbek
mogłoby zgubić głębię tła 1.0 i niesłusznie uznać obiekt za zasłonięty.
Wymiary 1×N, N×1 i 1×1 też są obsługiwane.

Usunięto pełnowymiarowy seed, oddzielny pass pierwszej redukcji, nieużywaną
teksturę minimum głębi, jej bindingi i zasoby odtwarzane podczas resize.
`hiz_size` przekazywane cullingowi odpowiada rzeczywistemu rozmiarowi nowej
podstawy. Liczbę mipów wylicza się przez całkowitoliczbowe `ilog2`.

Zachowano 4× MSAA i rozmiar workgroup 16×16. Zgodność liczby próbek jest
sprawdzana podczas kompilacji. Dobór 8×8 zamiast 16×16 wymagałby pomiaru na
konkretnej architekturze; nie zmieniano go bez takiego dowodu.

Usunięcie najdrobniejszego poziomu oznacza, że dla bardzo małych ekranowych
AABB culling może być mniej agresywny. Zredukowane maksimum pozostaje
konserwatywne: większy obszar może przepuścić dodatkową geometrię.
Pozostałe poziomy piramidy mają identyczne wartości z poprzednimi mipami 1+.

### Pomijanie zbędnej pracy i świeżość danych

- Resolve i generowanie mipów wykonują się tylko przy renderowaniu świata
  i włączonym cullingu okluzji.
- `hiz_valid` blokuje użycie piramidy do czasu jej pierwszego wygenerowania.
  Reset następuje przy resize, nowym świecie, seedzie z serwera, F9 oraz
  wyłączeniu tej ścieżki. Pierwsza klatka po ponownym włączeniu używa cullingu
  frustum, budując aktualną piramidę dla następnej klatki.
- Dotyczy to również resize, w którym dwa różne rozmiary powierzchni mają
  ten sam rozmiar zmniejszonej podstawy.
- Przy pominiętych etapach i podstawie 1×1 zapisywane są aktualne pary
  timestampów, aby HUD nie używał zapytań z poprzednich klatek. Gdy adapter
  nie obsługuje timestampów, nie są tworzone te puste passy.

## Pamięć i praca GPU

Oszczędność surowych danych texelowych dla typowych rozdzielczości wynosi
8 bajtów na piksel: pełny seed R32Float i nieużywane SSR depth R32Float.
Pozostała część piramidy zachowuje rozmiary. Tabela obejmuje wyłącznie Hi-Z
oraz dawną teksturę SSR depth, bez głównego depth MSAA i targetów koloru.

| Rozdzielczość | Przed zmianą | Po zmianie | Oszczędność |
|---|---:|---:|---:|
| 1920×1080 | 18,456 MiB | 2,636 MiB | 15,820 MiB |
| 2560×1440 | 32,812 MiB | 4,687 MiB | 28,125 MiB |
| 3840×2160 | 73,828 MiB | 10,546 MiB | 63,281 MiB |

To wyliczenie formatów i wymiarów; rzeczywista alokacja sterownika może
obejmować wyrównanie, tiling i metadane. Liczba wywołań shadera resolve
spada około czterokrotnie, lecz każde wykonuje redukcję większego obszaru.
Dlatego nie należy utożsamiać liczby invocationów ze wzrostem wydajności.
Usunięto jeden pass compute i zależność pamięciową między dwoma etapami.
Źródło wcześniejszego shadera zawierało powtórzone odczyty MSAA; kompilator
mógł je częściowo scalać, więc nie zakładano z góry dwukrotnego zmniejszenia
rzeczywistego ruchu pamięci.

## Weryfikacja wykonywanych shaderów

`tests/gpu_depth.rs` tworzy headless adapter Vulkan, rasteryzacją zapisuje
różne głębie dla każdej z czterech próbek MSAA, wykonuje compute i odczytuje
każdy mip. Wyniki porównuje jednocześnie z poprzednią ścieżką GPU oraz
niezależnym modelem redukcji CPU. Dawny resolve przechowywany jest wyłącznie
jako zamrożony fixture testowy.

Test przeszedł dla: 1×1, 1×17, 17×1, 2×2, 3×3, 16×16, 17×19, 31×32,
65×67 i 257×129. Obejmuje depth tła 1.0, różne wartości próbek, wymiary
niepodzielne przez workgroup, granice tekstur oraz wszystkie poziomy Hi-Z.

Dostępny adapter: **llvmpipe, LLVM 21.1.8, Mesa 26.0.8, Vulkan, device type
CPU**. Brak dostępu do sprzętowej karty przez `/dev/dri`. Test dowodzi
wykonalności oraz poprawności wyników na tym backendzie; nie zastępuje
interaktywnego testu obrazu na docelowym GPU.

Standardowa weryfikacja: `cargo test --offline --all-targets` — **41 testów
przeszło**, cztery testy/benchmarki wymagające jawnego uruchomienia pominięto.
Test wykonania GPU i benchmark timestampów uruchomiono dodatkowo i przeszły.
`cargo clippy --offline --all-targets` kończy się poprawnie z ostrzeżeniami
istniejących fragmentów; nowe pliki testów GPU nie zgłaszają ostrzeżeń.
`git diff --check` przeszedł.

## Mikrobenchmark timestampów

Pomiar obejmuje resolve oraz całą piramidę. Po rozgrzewce wykonuje 40 par
starej i nowej ścieżki, naprzemiennie zmieniając ich kolejność. Odczyt
wyników odbywa się po zakończeniu pomiaru. Podano mediany jednej próby:

| Rozmiar | Poprzednia ścieżka | Połączenie resolve + redukcja |
|---|---:|---:|
| 1280×720 | 3,000 ms | 1,624 ms |
| 1920×1080 | 8,287 ms | 3,950 ms |
| 1919×1079 | 9,632 ms | 6,294 ms |

**To wyniki programowego llvmpipe, nie pomiar FPS gry.** Potwierdzają poprawę
na dostępnym backendzie. Czasy i proporcje na sprzętowym GPU mogą się różnić;
nieparzyste wymiary wymagają więcej odczytów z powodu konserwatywnych
nakładających się obszarów. Nie mierzono docelowej sceny, poboru mocy ani
całego renderowania.

Odtwarzanie na docelowym sprzęcie, w dwóch osobnych uruchomieniach:

```bash
cargo test --offline --test gpu_depth fused_depth_matches_original_pyramid_and_cpu_reference -- --ignored --nocapture
cargo test --offline --test gpu_depth benchmark_depth_pyramid_gpu_timestamps -- --ignored --nocapture
```

Testy GPU są domyślnie ignorowane, ponieważ wymagają adaptera Vulkan.
Jawne uruchomienie bez adaptera lub wymaganych timestampów kończy się błędem,
zamiast udawać poprawne sprawdzenie.

## Pozostałe koszty i ograniczenia

1. **Temporal Hi-Z.** Culling używa piramidy poprzedniej klatki, a projekcji
   aktualnej kamery. `hiz_valid` chroni cykl życia zasobów; nie rozwiązuje
   reprojekcji podczas ruchu i odsłaniania wcześniej niewidocznych obszarów.
   Docelowo potrzebne są spójne macierze, konserwatywna obsługa disocclusion
   lub ograniczanie okluzji przy szybkich ruchach kamery.
2. **Niebo przed terenem.** Kosztowny proceduralny `sky.wgsl` jest rysowany
   przed opaque terrain, więc shader oblicza też piksele później przykryte
   geometrią. Przeniesienie sky na koniec opaque passu mogłoby wykorzystać
   early depth rejection, ale jego obecna głębia to 0.9999, nie 1.0. Taka
   zmiana wymaga poprawienia tej zależności i porównania obrazu dalekiego
   terenu; nie wprowadzono jej bez testu obrazu.
3. **Indirect.** Oba bufory komend są zerowane na klatkę. Przy
   `MULTI_DRAW_INDIRECT_COUNT` końcówka poza licznikiem widoczności nie jest
   odczytywana przez draw; osobna optymalizacja może pominąć jej czyszczenie.
   Fallback bez licznika nadal wymaga zerowania nieużywanych komend.
4. **Composite.** Każdy piksel wykonuje pięć próbek koloru dla bazowego obrazu
   i bloom; underwater dodaje dwie, menu blur następne szesnaście. Osobny
   bloom o mniejszej rozdzielczości lub warianty menu wymagają porównania
   jakości i czasu całego etapu, aby dodatkowe passy nie zniosły oszczędności.
5. **Rasteryzacja i MSAA.** Terrain, sky i transparent water nadal używają
   4× MSAA, a kolor resolve odbywa się po opaque i ponownie po wodzie.
   Opaque scene color jest potrzebny przez aktualny shader wody, więc jego
   resolve pozostawiono. Liczby próbek i jakości nie zmniejszano.
6. **Nieużywane zasoby.** Flow-map i część bindingów wody zostały po starszej
   implementacji, a eksperymentalny GPU mesher i sky upsample nie stanowią
   aktywnej ścieżki. Nie przypisywano im wpływu na zmierzoną wydajność gry.

W HUD należy porównywać sumę `depth_resolve_ms + hiz_ms`, ponieważ pierwsza
redukcja należy teraz do resolve. Sama zmiana czasu jednej z tych etykiet
nie pokazuje zysku całej ścieżki.

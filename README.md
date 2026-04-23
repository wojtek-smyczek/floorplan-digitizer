# Inwentaryzacja rzutu architektonicznego

Projekt konwertuje ręcznie rysowane uproszczone rzuty mieszkań i obiektów architektonicznych na pliki w formacie DXF, gotowe do otwarcia w programach CAD (AutoCAD, BricsCAD itp.).

Projekt realizowany w ramach kursu Pythona na Uniwersytecie Jagiellońskim.

## Proces tworzenia

**Zbieranie danych** — ręcznie narysowałem 63 uproszczonych rzutów zawierających trzy klasy obiektów (`wall`, `dimension_line`, `dimension_value`), a następnie zanotowałem je (bounding boxy + etykiety) w serwisie Roboflow.

**OCR cyfr** — przygotowałem własny dataset ~1000 odręcznie napisanych cyfr (po ~100 na cyfrę 0–9). Sam model dawał niezadowalające wyniki, więc połączyłem go z popularnym zbiorem MNIST (60 000 próbek) w dwufazowym treningu (pretrain + fine-tune), co znacząco poprawiło dokładność.

**Dopasowanie wymiarów do ścian** — pierwszym podejściem było korelowanie rozpoznanej wartości z linią wymiarową (`dimension_line`), a dopiero potem z elementem ściany. Po uzyskaniu średnich efektów przeszedłem do bezpośredniego dopasowania wartości wymiaru do ściany, z pominięciem linii wymiarowej. To uproszczenie przyniosło znacznie lepsze rezultaty.

## Narzędzia

Podczas developmentu korzystałem z Claude Code jako narzędzia wspomagającego — architektura pipeline'u, decyzje projektowe i zbieranie danych są mojego autorstwa.

## Instalacja + Szybkie Uruchomienie (generowanie rzutu rzut.dxf)

```bash
pip install -r requirements.txt
```

Wytrenuj lokalny model YOLO (wymaga datasetu w `yolo_dataset/` wyeksportowanego z Roboflow w formacie YOLOv8):

```bash
python train_yolo.py
cp runs/detect/yolo_floorplan/weights/best.pt yolo_floorplan.pt
```

Uruchomienie:

```bash
python main.py
```

## Opis działania

System składa się z trzech głównych modułów:

1. **`dataset_creator.py`** — tworzy zbiór danych treningowych z kart odręcznie napisanych cyfr
2. **`trainer.py`** — trenuje model sieci neuronowej (ResNet18) do rozpoznawania cyfr
3. **`train_yolo.py`** — trenuje lokalny model YOLOv8 do detekcji elementów rzutu
4. **`main.py`** — główny pipeline przetwarzający zdjęcie rzutu na plik DXF

Pipeline wykorzystuje:

- **YOLOv8 (Ultralytics, lokalnie)** — detekcja ścian, linii wymiarowych i wartości wymiarów na obrazie
- **Własny model CNN (ResNet18)** — rozpoznawanie odręcznie napisanych cyfr (OCR)
- **OpenCV** — przetwarzanie obrazu, segmentacja cyfr
- **ezdxf** — generowanie pliku CAD w formacie DXF

## Wymagania

- Python 3.10+
- Biblioteki wymienione w `requirements.txt`


## Struktura projektu

```

projekt-1-inwentaryzacja/
├── main.py # Główny pipeline (detekcja → OCR → DXF)
├── train_yolo.py # Trening lokalnego modelu YOLOv8
├── trainer.py # Skrypt trenowania modelu rozpoznawania cyfr
├── dataset_creator.py # Tworzenie datasetu z kart z cyframi
├── requirements.txt # Zależności Pythona
├── test.jpg # Przykładowy rzut do przetworzenia
├── digit_ocr_resnet18.pth # Wytrenowany model ResNet18 (~43 MB)
├── yolo_floorplan.pt # Wytrenowany model YOLOv8 (lokalny)
├── dane_projektu.json # Wynik detekcji YOLO (bounding boxy, klasy, wartości)
├── rzut.dxf # Wygenerowany plik CAD
│
├── sketches/ # Karty z odręcznymi cyframi (0.jpg–9.jpg)
├── my_dataset/ # Wycięte próbki cyfr (po ~100 na cyfrę)
│ ├── 0/ ... 9/
├── mnist_data/ # Dane MNIST (pobierane automatycznie)
├── debug_digits/ # Obrazy debugowe — wycięte cyfry z rzutu
└── roboflow_dataset/ # Odreczne rysowane rysunki rzutów

````

## Przygotowanie danych treningowych

### 1. Przygotowanie kart z cyframi

W folderze `sketches/` umieszczamy 10 zdjęć (pliki `0.jpg` do `9.jpg`), gdzie każde zdjęcie zawiera wiele instancji danej cyfry napisanej odręcznie na kartce.

### 2. Ekstrakcja próbek (`dataset_creator.py`)

```bash
python dataset_creator.py
````

Skrypt przetwarza każdą kartę:

- Konwertuje obraz do skali szarości i stosuje binaryzację (Otsu)
- Wykrywa kontury poszczególnych cyfr
- Filtruje za małe/za duże kontury
- Wycina i skaluje każdą cyfrę do rozmiaru 64×64 px
- Zapisuje próbki w `my_dataset/<cyfra>/`

Wynik: ~990 próbek treningowych (po ~100 na każdą cyfrę 0–9).

## Trenowanie modelu OCR

```bash
python trainer.py
```

Trening przebiega w dwóch fazach:

### Faza 1 — Pretrening

- Łączy własne dane z podzbiorem MNIST (5000 próbek)
- Własne dane mają 5× większą wagę niż MNIST
- 15 epok, Adam, lr=0.0005

### Faza 2 — Fine-tuning

- Trening wyłącznie na własnych danych
- 25 epok, lr=0.0001

**Architektura modelu:**

- Baza: ResNet18 z modyfikacjami
- Wejście: 1 kanał (obraz w skali szarości)
- Wyjście: 10 klas (cyfry 0–9)
- Regularyzacja: Dropout(0.5)
- Augmentacja danych: rotacja, pochylenie, perspektywa, rozmycie, zmiana jasności

Wynik: plik `digit_ocr_resnet18.pth`.

## Uruchomienie głównego pipeline'u

```bash
python main.py
```

Program przetworzy plik `test.jpg` i wygeneruje:

- `dane_projektu.json` — dane detekcji z rozpoznanymi wartościami
- `rzut.dxf` — plik CAD z odtworzonym rzutem
- `debug_digits/` — obrazy pomocnicze do weryfikacji OCR

## Opis pipeline'u krok po kroku

### Krok 1 — Detekcja obiektów (YOLO)

Na zdjęciu rzutu lokalny model YOLOv8 (Ultralytics) wykrywa trzy klasy obiektów:

- **Ściany** (`wall`) — prostokątne bounding boxy wokół narysowanych ścian
- **Linie wymiarowe** (`dimension_line`) — linie ze strzałkami wskazujące wymiar
- **Wartości wymiarów** (`dimension_value`) — obszary zawierające odręcznie napisane liczby

### Krok 2 — Postprocessing detekcji

- **NMS (Non-Maximum Suppression)** — usuwanie nakładających się detekcji
- **Łączenie sąsiednich wartości** — scalanie podzielonych detekcji wielocyfrowych liczb

### Krok 3 — Rozpoznawanie cyfr (OCR)

Dla każdej wykrytej wartości wymiaru:

1. Wyciągnięcie kanału czerwonego (wymiary rysowane na czerwono)
2. Binaryzacja progowa
3. Segmentacja na poszczególne cyfry (analiza projekcji kolumnowej)
4. Dopasowanie rozmiaru do 64×64 px
5. Klasyfikacja każdej cyfry przez model ResNet18
6. Filtracja po pewności (>70%)
7. Złożenie cyfr w pełną liczbę

### Krok 4 — Dopasowanie wymiarów do ścian

- Budowanie grafu ścian: podział na ściany pionowe (V) i poziome (H)
- Identyfikacja przerw (gaps) między sąsiednimi ścianami
- Przypisanie rozpoznanych wartości do odpowiednich przerw
- Estymacja skali (piksele → jednostki rzeczywiste)

### Krok 5 — Obliczanie współrzędnych

- Sekwencyjne wyznaczanie pozycji ścian na podstawie wymiarów
- Oś X: od lewej do prawej (na podstawie przerw H)
- Oś Y: od góry do dołu (na podstawie przerw V)

### Krok 6 — Domykanie rzutu

- Identyfikacja niepołączonych końców ścian
- Wykorzystanie nieprzypisanych wymiarów do wydłużenia ścian
- Rysowanie segmentów domykających

### Krok 7 — Eksport do DXF

Generowanie pliku CAD z odcinkami reprezentującymi ściany, kompatybilnego z AutoCAD i innymi programami CAD.

## Pliki wejściowe i wyjściowe

| Plik                 | Typ     | Opis                                       |
| -------------------- | ------- | ------------------------------------------ |
| `test.jpg`           | wejście | Zdjęcie/skan rzutu architektonicznego      |
| `digit_ocr_resnet18.pth`   | model   | Wytrenowany model ResNet18 do OCR          |
| `dane_projektu.json` | wyjście | Wyniki detekcji YOLO + rozpoznane wartości |
| `rzut.dxf`           | wyjście | Zrekonstruowany rzut w formacie CAD        |
| `debug_digits/`      | wyjście | Obrazy debugowe wycietych cyfr             |

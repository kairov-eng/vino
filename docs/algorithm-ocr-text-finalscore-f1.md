# Алгоритм сравнения текста этикетки (OCR ↔ каталог), FinalScore и F1

**Проект:** Vino Svoe (findwine)  
**Код:** `vino-svoe/backend/app/pipeline/ocr_match.py`  
**Версия алгоритма на момент примеров:** `0.29.1` (сканы 242–244)  
**Актуальная версия пайплайна:** см. `backend/app/pipeline/version.py` и обзор  
[`findwine-pipeline.md`](findwine-pipeline.md) (YOLO bottle+label, HSV, exclusive presence, XGB, CrEnc, выбор `matched_wine`)  
**Дата этого файла (OCR/TextScore слой):** 2026-09-21 · обновление ссылок: 2026-09-23  
**Сопровождающие данные:** `ocr-text-match-examples-242-244.json` (рядом с этим файлом)

> Этот документ детализирует **текстовый слой fin1** (TextScore / FinalScore / F1 / hard mismatch).  
> Полная схема поиска (детекция этикетки, HSV, exclusive lexicon, XGB, Cross Encoder, настройки) — в **findwine-pipeline.md**.

---

## 0. Преамбула: зачем это нужно и как вписывается в поиск

### Проблема

Пользователь фотографирует бутылку / этикетку. Система:

1. Находит **визуально похожие** вина в каталоге (embeddings SigLIP2 / опционально DINOv3) — top‑N кандидатов.
2. Параллельно читает текст с фото (**OCR**, часто Google Vision).
3. Должна выбрать **одно** вино (или честно сказать «не найдено»).

Только embedding недостаточно: у российских виноделен этикетки часто похожи (одна линейка, один стиль). Только OCR недостаточно: OCR ошибается, а в каталоге десятки «Красностоп» от разных хозяйств.

### Роль текстового сравнения

Текстовый слой отвечает на вопрос: **«Это OCR той же этикетки, что у карточки вина в каталоге?»**

Он:

- извлекает сущности из OCR (винодельня, название, цвет, тип, сорта, год);
- сравнивает их с полями карточки и с сохранённым `label` (текст этикетки в БД);
- выставляет **TextScore** (0..1);
- при жёстких противоречиях обнуляет кандидата (**hard mismatch**);
- считает **F1** по ключевым полям (метрика качества ключей, не замена FinalScore);
- вместе с **cosine** embedding даёт **FinalScore**; победитель — `argmax FinalScore` среди visual‑кандидатов (или отказ, если все отвергнуты).

### Где живёт результат

В `search_photos.status.steps.ocr_wine_id`:

- `per_variant.*.entities` — сущности OCR;
- `per_variant.*.visual_support[]` — скоринг кандидатов по каналу OCR;
- `final_ranked[]` — сводный ранг по всем каналам;
- `best_final` / `best_wine_id` — победитель;
- `ocr_rejected_all` — OCR отверг всех → `matched_wine` пустой (без fallback на top‑1 embedding).

UI показывает Text / Final / F1 на карточках кандидатов и в popup.

---

## 1. Входы

### 1.1. OCR искомого фото

Строка (с переносами строк) + опционально JSON LLM. Пример Google Vision:

```text
БЮРНЬЕ
BURNIER
КРАСНОСТОВ
```

### 1.2. Карточка вина в каталоге (`wines`)

| Поле | Назначение в матчинге |
|------|------------------------|
| `name` | название вина |
| `winery` | винодельня (для producer) |
| `color` / `category` | цвет / тип (сухое…) — эвристики + словари |
| `grape_variety` | купаж / сорта |
| `label` | сохранённый текст этикетки (многострочный) — **структура строк** |
| `region` | регион (мягкий вклад) |

Плюс **cosine** к query embedding (SigLIP2 предпочтительно; DINOv3 только если вина нет в SigLIP top‑N).

### 1.3. Пул кандидатов

`visual_ids` = unique(SigLIP top‑20 ∪ DINOv3 top‑20). OCR **не** ищет по всему каталогу — только по этому пулу.

---

## 2. Извлечение сущностей из OCR (`extract_entities` + merge с LLM JSON)

1. **Нормализация** (`normalize_ocr_text`): NFKC, ё→е, lowercase, вырезание шума (алкоголь %, 750 ml, «contains sulphites»…).
2. **Producer (винодельня):** матч строк OCR к списку `wines.winery`.  
   Важно: токены **сортов** (`пино`, `нуар`, `красностоп`…) **не** считаются доказательством винодельни (иначе «Пино Нуар» → ложное «Шато Пино»).
3. **Vintage:** год 19xx/20xx.
4. **color_key / type_key / grape_keys:** словари и эвристики.
5. **wine (residual):** нормализованный текст минус producer/region/vintage.
6. LLM JSON (если есть) мержится поверх text‑эвристик.

Пороги: `producer_score` обычно ≥ 70, чтобы producer участвовал в hard mismatch / F1.

---

## 3. TextScore (сравнение OCR‑сущностей с одной карточкой)

Для каждой пары (entities, wine) считаются similarity 0..100 (rapidfuzz) по полям, затем веса → **score ∈ [0,1]**.

### 3.1. Части (parts) — вклад 0..1

| Part | Как считается |
|------|----------------|
| `wine` | sim(OCR wine residual, name/label) |
| `producer` | sim(OCR producer, **только `winery`**, не label) + boost/штраф по пересечению non‑grape токенов |
| `color` | совместимость color_key ↔ catalog colors (100 / 0 / 50) |
| `type` | тип сухое/полусухое… (100 / 0 / 50) |
| `grape` | пересечение grape_keys (100 / 0 / 50) |
| `vintage` | soft: совпал год / другой год / отсутствует |
| `region`, `other`, `brand` | мягкие |

Веса полей задаются в настройках (в примерах 242–244: wine/producer/color/grape/type = 1.0, brand 0.12, …).

Базовая формула (внутри пайплайна **0..1**; в `status` / UI часто ×100, например `97.0`):

```text
TextScore_01 = Σ (parts[k] × w[k]) / Σ w[k]
TextScore_UI ≈ round(TextScore_01 × 100)
```

### 3.2. Hard mismatch → TextScore = 0 (и FinalScore = 0)

Если ключ **есть и в OCR, и в каталоге**, но sim &lt; `_KEY_CONFLICT_MAX` (45):

- `winery` (если producer уверенный),
- `name` (при отсутствии пересечения distinctive tokens),
- `color`, `type`, `grape`.

**Год не даёт hard mismatch** (то же вино другого урожая — только soft‑штраф).

### 3.3. Boost TextScore (0.92 / 0.97)

Если несколько ключей «присутствуют» и все прошли порог совпадения — score поднимают до 0.92 (2+ ключа) или 0.97 (3+).

С **0.30.0**: для учёта ключа `wine` в этом boost нужен `wine_sim ≥ 78` (раньше хватало 62 — из‑за этого «Бюрнье+сорт» уравнивали слабый wine_sim с полным совпадением label).

### 3.4. Структура строк OCR ↔ catalog `label` (`label_lines`)

Сравниваются **значимые строки** (raw OCR с `\n`, не сплющенный `text_norm`).

Строка «совпала», если fuzzy sim ≥ 65 **или** не конфликт краёв (первые 2 и последние 2 буквы).

**Слабая структура (`weak`)** примерно когда:

- сильно разное число строк **или** мало совпадений;
- несовпавшие строки «не похожи» по краям;
- matched_ratio низкий.

Если `weak` **и** `cosine < 0.8` → hard mismatch `label_lines` (точно другое вино при слабом visual).

При `cosine ≥ 0.8` это правило **не** режет (сильный visual может оправдать расхождение OCR/label).

---

## 4. FinalScore

```text
FinalScore = w_text_norm × TextScore_01 + w_cos_norm × Cosine
```

Веса из настроек нормализуются к сумме 1.  
В сканах 242–244: raw `text=0.5`, `cosine=0.8` → после нормализации ≈ **0.3846 / 0.6154**.  
(`TextScore_UI=97` → в формуле `0.97`.)

**Cosine:**

- берётся SigLIP2, если вино есть в SigLIP top‑N;
- иначе DINOv3 (если вино только там);
- **запрещён** `max(SigLIP, DINOv3)` — пространства несравнимы.

Hard mismatch → FinalScore = 0.

### 4.1. Выбор победителя

Сортировка `final_ranked`:

1. не hard mismatch и FinalScore &gt; 0;
2. больший FinalScore;
3. **(с 0.30.0)** больший OCR→label line recall (`matched / ocr_lines`);
4. больший `parts.wine`;
5. больший TextScore;
6. меньший id.

`matched_wine_id = best_final.id`.  
Если все кандидаты hard mismatch / FinalScore=0 → `ocr_rejected_all`, **без** подстановки top‑1 embedding.

---

## 5. F1 по ключевым полям

Ключи: `{winery, name, color, type, grape}` (**vintage не входит**).

Для каждого ключа:

- OCR «заявил» ключ (есть producer / wine tokens / color / type / grapes);
- каталог «имеет» ключ;
- **matched**, если обе стороны есть и sim ≥ `_KEY_MATCH_MIN` (62).

```text
tp = число matched ключей
fp = OCR заявил, но не matched
fn = каталог имеет, но не matched

P = tp / (tp + fp)     # precision
R = tp / (tp + fn)     # recall
F1 = 2PR / (P+R)       # 0 если hard_mismatch или P+R=0
```

F1 — **диагностическая** метрика согласованности ключей; победитель выбирается по FinalScore, не по F1.

На UI часто показывают F1×100.

---

## 6. Краткая схема потока

```text
Фото
  ├─ Embedding → top‑N wine ids + cosine
  └─ OCR → text (+json)
        → entities
        → для каждого wine_id из top‑N:
              TextScore + hard_mismatch + F1 + label_lines
              FinalScore = mix(TextScore, cosine)
        → final_ranked → best_final
        → если пусто: ocr_rejected_all (нет matched_wine)
```

---

## 7. Три примера из продакшен‑статуса (scanid 242, 243, 244)

Сырые выгрузки: `ocr-text-match-examples-242-244.json`.  
Алгоритм в статусе: **0.29.1**. Комментарии про 0.30.0 — как бы изменилось поведение после фикса «Classic».

---

### Пример A — scanid **242** (Senetkh / Красная Горка → «не найдено»)

#### OCR (Google Vision)

```text
SENETKH
Dba Dijbka
KPACHOCTON
KRASNOSTOP
Красная Горка
терруар
ограниченный выпуск
№ 009/800
2022
```

#### Сущности OCR

| Поле | Значение |
|------|----------|
| producer | *null* (Senetkh не сматчился к winery в каталоге достаточно уверенно / нет в списке) |
| wine residual | senetkh dba dijbka … красная горка … |
| color_key | `red` |
| grape_keys | [] (KRASNOSTOP не всегда попадает в grape_keys при латинице/OCR‑мусоре) |
| vintage | 2022 |

#### Что в каталоге у типичных кандидатов (top embedding)

Например id **2798** «АРАТТИ Рислинг…»:

- winery: АРАТТИ  
- color/category: белое  
- grape: Рислинг  
- label: `СЕМЕЙНАЯ ВИНОДЕЛЬНЯ / АРАТТИ / РИСЛИНГ / 2024 / ПОЛУСУХОЕ`  
- cos ≈ 0.72  

#### Как сравнил

| Проверка | Результат |
|----------|-----------|
| color | OCR `red` vs catalog white → **hard mismatch color** |
| name | токены Senetkh/… не пересекаются с АРАТТИ/Рислинг → **hard mismatch name** |
| TextScore / FinalScore | **0** |
| F1 | **0** (hard_mismatch) |

Аналогично другие кандидаты: hard mismatch по name/color/**label_lines** (слабая структура строк + cos &lt; 0.8).

#### Итог

```text
matched_wine_id = null
source = ocr_rejected_all
```

**Смысл:** OCR уверенно «не про» visual‑соседей → система **не** подставляет top‑1 embedding (это как раз чинит ложные «Красностоп Chateau Tamagne» на чужом Senetkh).

---

### Пример B — scanid **243** (Бюрнье Красностоп; победил 2909, эталон пользователя — 2910 Classic)

#### OCR

```text
БЮРНЬЕ
BURNIER
КРАСНОСТОв
```

(опечатка OCR: «КРАСНОСТОв» ≈ «КРАСНОСТОП»)

#### Сущности OCR

| Поле | Значение |
|------|----------|
| producer | Винодельня Бюрнье (score 100) |
| grape_keys | [`красностоп`] |
| wine residual | бюрнье burnier красностов |
| color/type | null |

#### Карточка победителя **2909** (как в status)

| Поле | Значение |
|------|----------|
| name | Бюрнье.Красностоп сухое красное |
| winery | Винодельня Бюрнье |
| label в БД | почти пустой: `Б Ю Р Н Ь Е` / `B U R N I E R` (без сорта!) |
| cosine | **0.7847** (#1 SigLIP) |
| TextScore | **97** (boost до 0.97 при wine_sim≈0.70) |
| FinalScore | **0.855975** ≈ 0.3846×0.97 + 0.6154×0.78471 |
| F1 | **0.75** (tp=3: winery, name, grape; fn=2: color/type в OCR нет → recall 0.6; P=1) |
| label_lines | ocr=3, matched=2, unmatched OCR: «красностов» (в label 2909 сорта нет) |

#### Ближайший правильный (по мнению пользователя) **2910** Classic

| Поле | Значение |
|------|----------|
| name | Бюрнье.Красностоп сухое красное **Classic** |
| winery | Винодельня Бюрнье |
| label | полный: Бюрнье / BURNIER / КРАСНОСТОП / … |
| cosine | 0.7617 |
| TextScore | 97 (тот же boost) |
| FinalScore | **0.842** |
| wine_sim (parts.wine) | **0.8531** vs **0.6951** у 2909 |
| label_lines | matched **3/3** OCR строк (OCR→label recall = 1.0; у 2909 = 2/3) |

#### Как сравнил и почему выиграл 2909 (в 0.29.1)

1. Оба: producer OK, grape красностоп OK → нет hard mismatch.  
2. Boost поднял TextScore обоих до 97, **смыв разницу wine_sim**.  
3. FinalScore ≈ 0.38×0.97 + 0.62×cos → выше cos у 2909 → **2909 победил**.  
4. Структура строк уже показывала, что label 2910 лучше кроет OCR, но в 0.29.1 это **не** влияло на ранг при равном TextScore.

#### Что меняет 0.30.0 (ожидаемо)

- Boost 0.97 только при wine_sim ≥ 78 → у 2909 TextScore ниже (нет полного boost).  
- Tie‑break: выше OCR→label recall → **2910**.  
Ожидаемые FinalScore порядка: 2909 ≈ 0.837, **2910 ≈ 0.842**.

---

### Пример C — scanid **244** (тот же Бюрнье OCR, повтор прогона)

#### OCR / сущности

Практически идентичны scan **243** (тот же текст Бюрнье / BURNIER / КРАСНОСТОВ).

#### Результат в status (0.29.1)

Снова **matched = 2909**, FinalScore 0.855975, сосед **2910** с 0.841844.

#### Зачем пример в тройке

Показывает **стабильность** ошибочного выбора при одинаковом OCR+пуле до фикса 0.30.0: проблема не «шум одного прогона», а систематический перекос FinalScore (cosine + одинаковый Text boost).

После 0.30.0 оба скана при пересчёте должны сходиться к **2910**, если visual‑пул и OCR те же.

---

## 8. Сводная таблица трёх сканов

| scanid | OCR (кратко) | matched (0.29.1) | Почему |
|--------|--------------|------------------|--------|
| **242** | Senetkh + KRASNOSTOP + Красная Горка | **null** (`ocr_rejected_all`) | Все visual‑кандидаты hard mismatch (цвет/имя/линии) |
| **243** | Бюрнье + Красностоп | **2909** (нужен 2910) | Один producer+grape; TextScore оба 97; выше cos у 2909 |
| **244** | то же | **2909** | Повтор той же логики |

---

## 9. Константы (важные пороги)

| Константа | Значение | Смысл |
|-----------|----------|--------|
| `_KEY_MATCH_MIN` | 62 | soft‑совпадение ключа / F1 match |
| `_KEY_CONFLICT_MAX` | 45 | ниже → hard mismatch |
| `_KEY_BOOST_WINE_MIN` | 78 (с 0.30.0) | wine_sim для boost 0.92/0.97 |
| `_LINE_STRUCT_COS_MAX` | 0.80 | ниже + weak lines → reject |
| `_LINE_STRUCT_MATCH_MIN` | 65 | sim строки «совпала» |

Веса Text/Final — runtime settings (в примерах final text/cos = 0.5/0.8).

---

## 10. Как читать JSON для другой LLM

Файл `ocr-text-match-examples-242-244.json` — массив из трёх объектов:

- `ocr_google_vision_text` — сырой OCR;
- `ocr_entities` — извлечённые поля;
- `matched_wine` — решение пайплайна;
- `candidates_detail[]` — карточка + parts + F1 + label_lines + hard_mismatch для топ‑кандидатов (включая 2910 на 243/244);
- `final_weights` / `text_weights` — фактические веса прогона;
- `siglip_top5` — visual top без текста.

Рекомендуемый запрос к другой модели:  
«По формуле из algorithm-ocr-text-finalscore-f1.md пересчитай FinalScore для candidates_detail scan 243 и скажи, кто победит при правилах 0.30.0».

---

## 11. Ограничения (честно)

1. OCR может не прочитать «Classic» / год / цвет — тогда текст не отличит SKU одной линейки.  
2. Плохой/пустой `label` в каталоге ломает line‑структуру (как у 2909).  
3. F1 ≠ качество «это то вино»; высокий F1 возможен при неверной SKU той же линейки.  
4. Пул = только embedding top‑N: правильное вино вне top‑N невосстановимо текстом.  
5. Документ описывает логику кода; конкретные веса на деплое могут отличаться от дефолтов `.env`.

---

*Конец документа. Предназначен для передачи другим LLM / ревьюерам без доступа к IDE.*

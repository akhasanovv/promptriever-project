# Promptriever RS

Каркас исследовательского проекта для русскоязычного воспроизведения идей Promptriever:

- сборка кастомного обучающего датасета из `SberQuAD`,
- генерация негативных инструкций через `Llama-8B` по `Groq API`,
- дообучение retriever-моделей, начиная с `multilingual-e5-base`,
- оценка на `ruMTEB` и `mFollowIR`,
- быстрый запуск новых baseline-моделей и конфигураций через `yaml`.

## Почему структура устроена именно так

Из текста вашей работы следуют три независимых исследовательских контура:

1. `Instruction-following quality`
   Нужно учить модель реагировать на инструкцию, а не только на смысл запроса.
2. `Retrieval quality`
   Нельзя улучшить `pMRR` ценой полного развала `nDCG@10`.
3. `Experiment management`
   Нужно удобно сравнивать модели, шаблоны форматирования, типы негативов, loss-функции и наборы бенчмарков.

Поэтому проект разделён на модули `data / generation / training / evaluation / models`, а все запускаемые эксперименты описываются конфигами.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[train,eval,dev]'
```

Если окружение уже было установлено раньше, обновите зависимости:

```bash
pip install -r requirements.txt
pip install -e '.[train,eval,dev]'
```

Проверка CLI:

```bash
promptriever-rs --help
```

## Базовый workflow

1. Подготовить данные SberQuAD:

```bash
promptriever-rs data build-sberquad \
  --config configs/dataset/sberquad_base.yaml
```

2. Сгенерировать запросы к Llama-8B через Groq:

```bash
export GROQ_API_KEY=...
promptriever-rs generation generate-negatives \
  --config configs/dataset/sberquad_negative_generation.yaml
```

3. Собрать train/val dataset:

```bash
promptriever-rs data assemble-training-set \
  --config configs/dataset/sberquad_instruction_pairs.yaml
```

4. Дообучить baseline `multilingual-e5-base`:

```bash
promptriever-rs train fit \
  --config configs/train/e5_instruction_pairs.yaml
```

5. Прогнать оценку:

```bash
promptriever-rs eval rumteb \
  --config configs/eval/rumteb_russian.yaml

promptriever-rs eval mfollowir \
  --config configs/eval/mfollowir_russian.yaml
```

## Основные директории

- `configs/` — конфиги датасетов, моделей, тренировки и оценки.
- `src/promptriever_rs/` — Python-пакет.
- `data/raw` — выгруженные исходные наборы.
- `data/interim` — промежуточные jsonl/csv артефакты.
- `data/processed` — финальные train/val/test наборы.
- `outputs/` — чекпоинты, метрики, manifests экспериментов.

## Формат основного train-сэмпла

Финальный jsonl датасет рассчитан на несколько режимов обучения сразу:

```json
{
  "sample_id": "train-000001",
  "split": "train",
  "query": "Кто написал роман ...?",
  "answer": "Иванов",
  "positive_passage": "Контекст из SberQuAD ...",
  "positive_instruction": "Найди документ, который точно отвечает на вопрос.",
  "negative_instruction": "Найди документ по той же теме, но без точного ответа на вопрос.",
  "negative_passages": [],
  "metadata": {
    "source_dataset": "kuznetsoffandrey/sberquad",
    "context_id": "ctx-001234"
  }
}
```

Это позволяет:

- учить только на позитивных инструкциях,
- учить бинарные пары `(instruction + query, passage)` с метками `1/0`,
- позже добавлять mined hard negatives по passages без изменения общего формата.

## Что уже учтено под вашу постановку

- Русскоязычный `SberQuAD` как источник `(query, text)` пар.
- По умолчанию сохраняются все пары `question -> context`, а дедупликация контекстов оставлена только как опциональный режим.
- `multilingual-e5-base` как baseline.
- Форматные префиксы `query:` / `passage:` для E5.
- Разделение `ruMTEB` и `mFollowIR` в отдельные evaluators.
- Генерация негативных инструкций через `Groq API` без привязки к конкретной SDK.
- Возможность быстро подключить `bge-m3` и другие эмбеддеры через registry.

## Следующие естественные шаги

- Подключить hard-negative mining по исходному retriever.
- Добавить отдельный loss для instruction discrimination.
- Сделать sweep по шаблонам инструкций и форматам query input.

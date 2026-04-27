# Promptriever

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

2. Сгенерировать синтетические инструкции с помощью LLM:

Интерфейс для Groq:

```bash
export GROQ_API_KEY=...
promptriever-rs generation generate-positives \
  --config configs/dataset/sberquad_positive_generation.yaml

promptriever-rs generation generate-negatives \
  --config configs/dataset/sberquad_negative_generation.yaml
```

Интерфейс для Openrouter:

```bash
export OPENROUTER_API_KEY=...
promptriever-rs generation generate-negatives \
  --config configs/dataset/sberquad_negative_generation_openrouter.yaml
```

Чтобы возобновить генерацию с произвольного места, выставьте в конфиге:

```yaml
resume: true
start_index: 12000
```

Тогда генератор начнет просмотр входного `jsonl` с индекса `12000` и дополнительно пропустит все `sample_id`, которые уже есть в выходном файле.

3. Собрать train/val dataset:

```bash
promptriever-rs data assemble-training-set \
  --config configs/dataset/sberquad_instruction_pairs.yaml
```

Если хотите обучаться не на фиксированной позитивной инструкции, а на LLM-сгенерированных позитивных инструкциях, используйте:

```bash
promptriever-rs data assemble-training-set \
  --config configs/dataset/sberquad_instruction_pairs_positive_llm.yaml
```

4. Дообучить baseline `multilingual-e5-base`:

```bash
promptriever-rs train fit \
  --config configs/train/e5_instruction_pairs.yaml
```

Вариант с LLM-сгенерированными позитивными инструкциями:

```bash
promptriever-rs train fit \
  --config configs/train/e5_instruction_pairs_positive_llm.yaml
```

В train/eval конфигах поддерживается выбор устройства:

- `device: auto` — сначала `cuda`, потом `mps`, иначе `cpu`
- `device: cuda`
- `device: mps`
- `device: cpu`

Для Apple Silicon используйте `device: mps`. Для NVIDIA GPU — `device: cuda`.
Параметр `use_fp16: true` имеет смысл только на `cuda`; на `mps` и `cpu` он автоматически отключается.

Обучение по умолчанию идет через LoRA (`use_lora: true`). Гиперпараметры LoRA задаются прямо в train-конфиге:

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [query, key, value]
  bias: none
```

Для экспериментов меняйте `lora.r`, `lora.alpha`, `lora.dropout`, `batch_size`, `learning_rate` и `num_epochs` в yaml-файле без изменения кода.

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

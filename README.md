# Promptriever

Репозиторий исследовательского проекта по статье Promptriever. 

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[train,eval,dev]'
```

Проверить, что CLI доступен:

```bash
promptriever-rs --help
```

## Структура

- `configs/dataset/` - конфиги сборки датасета и генерации LLM-артефактов
- `configs/train/` - конфиги обучения
- `configs/eval/` - конфиги оценки
- `configs/pipeline/` - общий пайплайн эксперимента
- `configs/models/` - конфиги базовых моделей
- `src/promptriever_rs/` - код пакета
- `data/processed/` - финальный датасет для обучения
- `docs/` - исследовательские заметки

Промежуточные данные ожидаются в `data/interim/`, а результаты обучения и оценки - в `outputs/`.

## Основной пайплайн

Финальный workflow лежит в:

```text
configs/pipeline/e5_full_promptriever.yaml
```

### 1. Базовые записи SberQuAD

```bash
promptriever-rs data build-sberquad \
  --config configs/dataset/sberquad_base.yaml
```

На этом шаге исходный SberQuAD приводится к единому JSONL-формату с `sample_id`, `split`, `query`, `answer` и `positive_passage`.

### 2. Генерация позитивных инструкций

```bash
export GROQ_API_KEY=...

promptriever-rs generation generate-positives \
  --config configs/dataset/sberquad_positive_generation.yaml
```

Позитивная инструкция строится из пары `query + positive_passage`. Она должна уточнять поисковую задачу, но не ломать релевантность исходного passage.

### 3. Валидация позитивных инструкций

```bash
promptriever-rs generation validate-positives \
  --config configs/dataset/sberquad_positive_instruction_validation.yaml
```

Judge-модель проверяет, что `instruction + query` всё еще соответствует исходному passage.

Если нужно поменять только порог валидности, можно не запускать judge заново:

```bash
promptriever-rs generation apply-positive-thresholds \
  --config configs/dataset/sberquad_positive_instruction_thresholds.yaml
```

### 4. Поиск hard negatives

```bash
promptriever-rs data mine-hard-negatives \
  --config configs/dataset/sberquad_hard_negatives.yaml
```

Для каждого запроса извлекаются кандидаты из корпуса, после чего judge отфильтровывает passages, которые не отвечают на исходный `query`. В итоговый датасет попадают hard negatives, полезные для обучения ретривера.

Для частичного запуска можно указать в конфиге:

```yaml
start_index: 0
max_samples: 5000
```

### 5. Генерация Promptriever-style passages

```bash
promptriever-rs generation generate-passages \
  --config configs/dataset/sberquad_promptriever_passages.yaml
```

LLM генерирует:

- `generated_positive_passage` - passage, релевантный полному `query + instruction`;
- `instruction_negative_passages` - passages, которые отвечают на базовый `query`, но не удовлетворяют инструкции.

### 6. Валидация сгенерированных passages

```bash
promptriever-rs generation validate-passages \
  --config configs/dataset/sberquad_promptriever_passage_validation.yaml
```

Проверяются три условия:

1. `generated_positive_passage` подходит к `query + instruction`;
2. каждый instruction-negative passage подходит к базовому `query`;
3. тот же instruction-negative passage не подходит к полному `query + instruction`.

Пороги можно пере-применить без повторного judge-запуска:

```bash
promptriever-rs generation apply-passage-thresholds \
  --config configs/dataset/sberquad_promptriever_passage_thresholds.yaml
```

### 7. Сборка финального датасета

```bash
promptriever-rs data assemble-promptriever-set \
  --config configs/dataset/sberquad_promptriever_dataset.yaml
```

На выходе получается:

```text
data/processed/sberquad_promptriever.jsonl
```

### 8. Обучение

```bash
promptriever-rs train fit \
  --config configs/train/e5_promptriever.yaml
```

### 9. Оценка

```bash
promptriever-rs eval rumteb \
  --config configs/eval/rumteb_russian.yaml

promptriever-rs eval mfollowir \
  --config configs/eval/mfollowir_russian.yaml
```

## Возобновление генерации

Генерацию можно продолжать с нужного места:

```yaml
resume: true
start_index: 12000
```

При `resume: true` генератор читает входной JSONL с `start_index` и дополнительно пропускает `sample_id`, которые уже есть в выходном файле.

## Конфигурация обучения

Train- и eval-конфиги поддерживают:

- `device: auto` - сначала `cuda`, затем `mps`, иначе `cpu`
- `device: cuda`
- `device: mps`
- `device: cpu`

Чтобы использовать LoRA:

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - query
    - key
    - value
  bias: none
```

Гиперпараметры, которые обычно имеет смысл варьировать:

- `lora.r`
- `lora.alpha`
- `lora.dropout`
- `batch_size`
- `learning_rate`
- `num_epochs`
- `negatives_per_sample`

## Формат финального датасета

```json
{
  "sample_id": "train-000001",
  "split": "train",
  "query": "Кто написал роман ...?",
  "answer": "Иванов",
  "instruction": "Найди документ, который отвечает на вопрос и дополнительно указывает год публикации.",
  "positive_passage": "Подходящий passage, удовлетворяющий и вопросу, и инструкции ...",
  "instruction_negative_passages": [
    "Пассаж отвечает на базовый вопрос, но не содержит год публикации ...",
    "Пассаж по той же теме, но с нарушением условия инструкции ..."
  ],
  "hard_negative_passages": [
    "Пассаж из корпуса, не отвечающий на исходный запрос ...",
    "Еще один hard negative passage ..."
  ],
  "metadata": {
    "source_dataset": "kuznetsoffandrey/sberquad",
    "context_id": "ctx-001234",
    "generated_positive_score": 3.42,
    "hard_negative_scores": [-1.27, -0.84]
  }
}
```

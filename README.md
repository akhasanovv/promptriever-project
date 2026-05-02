# Promptriever

Исследовательский код для instruction-aware dense retrieval на русскоязычных данных, вдохновленный статьей Promptriever и адаптированный для экспериментов на основе SberQuAD.

## Обзор

Проект поддерживает два режима обучения:

- `Promptriever-style` обучение, где anchor задается как `query + instruction`, а негативы представлены явными `instruction-negative passages`
- `Pairwise baseline` обучение, где один и тот же passage сочетается с `positive_instruction` и `negative_instruction`

Основной workflow в репозитории построен вокруг Promptriever-style режима.

## Установка

Создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[train,eval,dev]'
```

Если окружение уже существует, обновите зависимости:

```bash
pip install -r requirements.txt
pip install -e '.[train,eval,dev]'
```

Проверка CLI:

```bash
promptriever-rs --help
```

## Структура репозитория

- `configs/` - конфигурации датасетов, моделей, обучения, оценки и пайплайнов
- `src/promptriever_rs/` - исходный код Python-пакета
- `data/raw/` - исходные загруженные данные
- `data/interim/` - промежуточные JSONL- и CSV-артефакты
- `data/processed/` - финальные train-, validation- и test-датасеты
- `outputs/` - чекпоинты, метрики и manifests экспериментов
- `docs/` - исследовательские заметки и проектная документация

## Основной пайплайн

Рекомендуемый workflow воспроизводит основную идею Promptriever настолько близко, насколько это возможно в рамках текущего кодового каркаса.

### 1. Подготовка базовых записей SberQuAD

```bash
promptriever-rs data build-sberquad \
  --config configs/dataset/sberquad_base.yaml
```

### 2. Генерация позитивных инструкций

Позитивные инструкции генерируются из `query + positive_passage`, чтобы исходный passage оставался релевантным после добавления инструкции.

```bash
export GROQ_API_KEY=...
promptriever-rs generation generate-positives \
  --config configs/dataset/sberquad_positive_generation.yaml
```

### 3. Валидация позитивных инструкций

Для правильной версии датасета позитивные инструкции дополнительно фильтруются judge-моделью, которая проверяет, что `query + instruction` по-прежнему соответствует исходному passage.

```bash
promptriever-rs generation validate-positives \
  --config configs/dataset/sberquad_positive_instruction_validation.yaml
```

Если нужно только поменять порог валидности, повторно запускать judge не требуется. Достаточно пере-применить threshold к уже сохраненным `validation_score`:

```bash
promptriever-rs generation apply-positive-thresholds \
  --config configs/dataset/sberquad_positive_instruction_thresholds.yaml
```

### 4. Поиск hard negatives по query

Для каждого запроса извлекаются top-k кандидаты из векторной базы, после чего judge-модель отфильтровывает passages, не отвечающие на исходный `query`. В итоговый датасет сохраняются 2 таких passage.

```bash
promptriever-rs data mine-hard-negatives \
  --config configs/dataset/sberquad_hard_negatives.yaml
```

Чтобы искать hard negatives только для части датасета, можно использовать:

```yaml
start_index: 0
max_samples: 5000
```

В этом случае hard negatives будут найдены только для первых `5000` примеров, начиная с `start_index`. Корпус passages при этом останется полным.

### 5. Генерация Promptriever-style passages

На этом шаге создаются:

- `generated_positive_passage`, релевантный и запросу, и инструкции
- `instruction_negative_passages`, релевантные базовому запросу, но не удовлетворяющие полному `query + instruction`

```bash
promptriever-rs generation generate-passages \
  --config configs/dataset/sberquad_promptriever_passages.yaml
```

### 6. Валидация сгенерированных passages

Judge-модель проверяет:

- что `generated_positive_passage` подходит к `query + instruction`
- что каждый `instruction_negative_passage` подходит к `query`, но не подходит к `query + instruction`

```bash
promptriever-rs generation validate-passages \
  --config configs/dataset/sberquad_promptriever_passage_validation.yaml
```

Если требуется изменить пороги для:

- `generated_positive_is_valid`
- соответствия `instruction_negative_passages` базовому `query`
- несоответствия `instruction_negative_passages` полному `query + instruction`

можно выполнить быструю пере-разметку по уже сохраненным score-значениям без повторного запуска judge:

```bash
promptriever-rs generation apply-passage-thresholds \
  --config configs/dataset/sberquad_promptriever_passage_thresholds.yaml
```

### 7. Сборка обучающего датасета

```bash
promptriever-rs data assemble-promptriever-set \
  --config configs/dataset/sberquad_promptriever_dataset.yaml
```

### 8. Обучение ретривера

```bash
promptriever-rs train fit \
  --config configs/train/e5_promptriever.yaml
```

В этом режиме:

- anchor: `query + instruction`
- positive: `positive_passage`
- negatives: `instruction_negative_passages` и `hard_negative_passages`
- loss: `MultipleNegativesRankingLoss`

### 9. Запуск оценки

```bash
promptriever-rs eval rumteb \
  --config configs/eval/rumteb_russian.yaml

promptriever-rs eval mfollowir \
  --config configs/eval/mfollowir_russian.yaml
```

## Baseline-пайплайн

Репозиторий также содержит более простой pairwise baseline для ablation-экспериментов.

```bash
export GROQ_API_KEY=...
promptriever-rs generation generate-positives \
  --config configs/dataset/sberquad_positive_generation.yaml

promptriever-rs generation generate-negatives \
  --config configs/dataset/sberquad_negative_generation.yaml

promptriever-rs data assemble-training-set \
  --config configs/dataset/sberquad_instruction_pairs_positive_llm.yaml

promptriever-rs train fit \
  --config configs/train/e5_instruction_pairs_positive_llm.yaml
```

Также доступна конфигурация для OpenRouter:

```bash
export OPENROUTER_API_KEY=...
promptriever-rs generation generate-negatives \
  --config configs/dataset/sberquad_negative_generation_openrouter.yaml
```

## Возобновление генерации

Генерацию можно продолжать с произвольного смещения с помощью следующих полей в конфиге:

```yaml
resume: true
start_index: 12000
```

При возобновлении генератор начинает читать входной JSONL с `start_index` и дополнительно пропускает все значения `sample_id`, которые уже присутствуют в выходном файле.

## Конфигурация обучения

Train- и eval-конфиги поддерживают следующие варианты устройства:

- `device: auto` - сначала `cuda`, затем `mps`, иначе `cpu`
- `device: cuda`
- `device: mps`
- `device: cpu`

Для Apple Silicon рекомендуется `device: mps`. Для NVIDIA GPU - `device: cuda`.
Параметр `use_fp16: true` используется только на CUDA и автоматически отключается на `mps` и `cpu`.

По умолчанию включен LoRA:

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [query, key, value]
  bias: none
```

Типичные гиперпараметры для варьирования между экспериментами:

- `lora.r`
- `lora.alpha`
- `lora.dropout`
- `batch_size`
- `learning_rate`
- `num_epochs`

## Формат датасета

### Promptriever-style датасет

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

### Pairwise baseline датасет

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

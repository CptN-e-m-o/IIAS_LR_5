# ЛР5: эксперименты по маршрутизации helpdesk-тикетов

Программа проверяет baseline-гипотезу для ЛР5: TF-IDF-признаки из текста обращения позволяют классифицировать тикеты по очередям `queue` лучше наивного ориентира `Dummy most frequent`.

## Датасет

Используется датасет `Tobi-Bueck/customer-support-tickets` с Hugging Face. В архив уже положен CSV-файл:

`data/customer_support_tickets.csv`

По умолчанию эксперименты запускаются только по англоязычной части датасета (`language=en`), чтобы TF-IDF с английскими stop words соответствовал данным.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

```bash
./run.sh
```

Или напрямую:

```bash
python src/run_experiments.py --data data/customer_support_tickets.csv --out outputs --language en
```

Для запуска на всех языках:

```bash
python src/run_experiments.py --data data/customer_support_tickets.csv --out outputs --language all
```

## Что создаётся в `outputs/`

- `metadata.json` — параметры эксперимента, split, классы, seed.
- `runs.csv` — результаты каждого запуска.
- `summary.csv` — средние значения, 95% CI и p-value.
- `boxplot_macro_f1.png` — boxplot по macro-F1.
- `learning_curve.png` — кривая обучения.
- `confusion_matrix.png` — нормализованная матрица ошибок.
- `pr_curve.png` — PR-кривая.
- `error_examples.csv` — примеры ошибок.
- `classification_report_subject_body.csv` — отчет по классам для основного baseline.

## Проверяемая гипотеза

Применение TF-IDF-признаков из текста обращения к задаче маршрутизации helpdesk-тикетов по очередям `queue` позволит повысить macro-F1 не менее чем на 20 п.п. по сравнению с `Dummy most frequent` за счёт учета лексических различий между типами заявок.

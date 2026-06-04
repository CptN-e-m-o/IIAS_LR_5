# Лабораторная работа №5

Тема: описание и обсуждение результатов экспериментов для задачи автоматической маршрутизации helpdesk-тикетов.

## Данные

Используется датасет `Tobi-Bueck/customer-support-tickets`.
Скачайте файл `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, переименуйте его в `customer_support_tickets.csv` и положите в папку `data/`.

Структура:

```text
project/
├── data/
│   └── customer_support_tickets.csv
├── lr5_experiments.py
├── requirements.txt
└── outputs/
```

## Установка

```bash
pip install -r requirements.txt
```

## Быстрая проверка

```bash
python lr5_experiments.py --data data/customer_support_tickets.csv --output-dir outputs --quick
```

## Полный запуск

```bash
python lr5_experiments.py --data data/customer_support_tickets.csv --output-dir outputs
```

## Что делает скрипт

- запускает 7 повторных экспериментов с фиксированными seed;
- сравнивает основной подход с ориентирами и абляциями;
- считает accuracy, macro-F1, weighted-F1;
- рассчитывает 95% CI и paired t-test;
- строит boxplot, confusion matrix, PR-кривую и learning curve;
- сохраняет примеры ошибок.

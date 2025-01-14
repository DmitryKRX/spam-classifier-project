<a href="https://colab.research.google.com/drive/10ZfVn7CXRQiJ6pR5JY4yOe13nhm1pYIE">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

# Анти-спам классификатор

Данный проект подразумевает создание классификатора на базе моделей машинного обучения для выявления спам-сообщений.

## Датасет
Для обучения моделей бинарной классификации сообщений (спам/не спам) на русском языке использовался датасет, собранный из общедоступных данных в Интернете. Часть данных была на английском языке и, соответственно, переведена на русский язык с использованием библиотеки Deep-Translator. Датасет состоит примерно из 66000 записей с меткой класса. 

Ссылка на датасет: https://huggingface.co/datasets/DmitryKRX/anti_spam_ru

## Результаты

Для оценки качества используется F1 мера.

Обучены следующие модели и их метрики:

- TF-IDF + Логистическая регрессия: F1 = 0.9333;
- TF-IDF + Random Forest: F1 = 0.9503
- TF-IDF + CatBoost: F1 = 0.9345
- LSTM: F1 = 0.9669
- RuBertTiny: F1 = 0.9020

Наилучшие показатели у LSTM-модели. Однако все модели, кроме RuBertTiny, имеют словарь, ограничивающийся набормо слов и датасета, следовательно, будут плохо показывать себя на инференсе, чего нельзя сказать про RuBertTiny модель.

Для инференса была выбрана модель на основе RuBert Tiny.

## Примеры

<img src="https://github.com/user-attachments/assets/8d47c2ed-ef95-451c-8f8d-771591b627ba" alt="Пример работы бота" width="568" height="653">

---

<img src="https://github.com/user-attachments/assets/9ad35217-f674-40d2-b3c6-6a8597ef8c0a" alt="Пример работы бота" width="554" height="272">

## Описание файлов

- training.ipynb — блокнот с обработкой данных и обучением всех моделей;
- st_spam_classifier.py — файл с инференсом модели RuBertTiny в интерфейсе Streamlit;
- rubert_tiny_model — сохраненная модель RuBertTiny из библиотеки transformers;
- rubert_tiny_tokenizer — сохраненный токенизатор RuBertTiny из библиотеки transformers;
- rubert_classifier_weights.pth — файл с весами классификатора из модели на базе RuBertTiny;
- requirements.txt — список зависимостей и их версий.


## Запуск:

```
python -m venv venv
```
- Windows:
  ```
  .\venv\Scripts\activate
  ```
- MacOS/Linux
  ```
  source venv/bin/activate
  ```

```
pip install -r requirements.txt
```
```
python -m streamlit run st_spam_classifier.py
```

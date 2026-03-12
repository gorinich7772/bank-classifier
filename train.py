import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Настройки
model_name = "deepvk/RuModernBERT-base" # Новая мощная архитектура
num_labels = 7 # Количество твоих тем

# 2. Подготовка данных (замени на свой файл)
def prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    # Кодируем названия тем в числа (0-6)
    label2id = {
        "Кредитование": 0, "Ипотека": 1, "Вклады": 2,
        "Обмен валют": 3, "Зарплатные проекты": 4,
        "Онлайн-банкинг": 5, "Страхование": 6
    }
    df['label'] = df['label'].map(label2id)
    return Dataset.from_pandas(df)

# 3. Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 4. Основной процесс
dataset = prepare_data("banking_dataset.csv")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32, #сколько примеров за один раз модель «прочитывает»
    num_train_epochs=100,
    weight_decay=0.01,
    # evaluation_strategy="no", # Можно добавить валидацию
    save_strategy="epoch",
    fp16=True, # Включаем, если у тебя карта RTX
    tf32=True,                       # Ускоряет матричные вычисления на Tensor Cores 5-го поколения
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Поехали!
trainer.train()
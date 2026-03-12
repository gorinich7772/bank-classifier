import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Путь к твоей обученной модели (замени на актуальный чекпоинт из папки results)
model_path = "./results/checkpoint-500"  # или просто "./results" в конце обучения
tokenizer_name = "deepvk/RuModernBERT-base"

# 2. Загрузка
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Переводим в режим предсказания

# Список меток (должен быть в том же порядке, что и при обучении)
labels = ["Кредитование", "Ипотека", "Вклады", "Обмен валют",
          "Зарплатные проекты", "Онлайн-банкинг", "Страхование"]


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()

    print(f"Текст: {text}")
    print(f"Категория: {labels[prediction]} (уверенность: {probabilities[0][prediction]:.2%})")


# Пробуем!
print("-" * 30)
predict("Я потерял доступ к приложению и не могу войти")
predict("Хочу положить деньги под высокий процент на полгода")
predict("Застрахуйте мою квартиру от пожара")
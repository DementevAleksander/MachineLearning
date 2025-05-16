from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import gradio as gr

# Загрузка модели и токенизатора
model_path = "/app/fine-tuned-ruBert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Словарь меток
label_map = {
    0: "Ошибка/Проблема",
    1: "Консультации",
    2: "Предложения по улучшению",
    3: "Оборудование"
}


# Функция предсказания
def predict(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return label_map[predicted_class]


# Интерфейс Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Введите ваш запрос...", label="Текст обращения"),
    outputs=gr.Label(label="Предсказанный класс"),
    title="Классификация обращений",
    description="Введите текст обращения, и система определит его тип."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
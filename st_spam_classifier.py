import streamlit as st
import torch
import torch.nn as nn
import warnings
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# Класс модели
class TransformerClassificationModel(nn.Module):
    def __init__(self, base_transformer_model: nn.Module, hidden_size=256):
        super(TransformerClassificationModel, self).__init__()
        self.backbone = base_transformer_model

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        output = output.last_hidden_state[:, 0]
        output = self.classifier(output)
        return output


# Загрузка и настройка модели
model = TransformerClassificationModel(AutoModel.from_pretrained("rubert_tiny_model"))
model.classifier.load_state_dict(torch.load('rubert_classifier_weights.pth', weights_only=True))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

tokenizer = AutoTokenizer.from_pretrained('rubert_tiny_tokenizer')

# Функция кодирования текста
def vectorizer(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}

# Функция для классификации текста
def bert_classify(inputs):
    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)).reshape(-1).cpu()
    return outputs


# Функция для обработки сообщения
def process_message(text):
    embeddings = vectorizer(text)
    prediction = bert_classify(embeddings)
    return f"Спам ({(prediction[0]*100):.2f}%)" if prediction >= 0.5 else f"Не спам ({(prediction[0]*100):.2f}%)"

### Настройка Streamlit интерфейса
st.header("Чат-бот для классификации спам-сообщений")

# Стартовое сообщение
start = 'Привет! Отправь мне текст, и я скажу, спам это или нет (и в скобках отмечу вероятность текста быть спамом).'
with st.chat_message("assistant"):
    st.markdown(start)
    
# Состояние чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Ввод сообщения пользователем
if msg := st.chat_input("Напишите сообщение"):
    # Сохраняем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": msg})
    with st.chat_message("user"):
        st.markdown(msg)
    
    # Ответ от модели
    response = process_message(msg)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

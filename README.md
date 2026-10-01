# Amy — пробный чат с нейросетью в стиле Character.ai

Одностраничный сайт-переписка с AI-персонажем **Amy (Эми)**: ей 23 года,
она работает менеджером в отеле. Интерфейс сделан в духе Character.ai
(карточка персонажа сбоку, пузыри диалога, «печатание» ответа, быстрые
подсказки).

## Публикация на GitHub

Код уже закоммичен в ветке. Чтобы опубликовать репозиторий:

1. Создайте пустой публичный репозиторий на GitHub (например, `amy-chat`).
2. Выполните в папке проекта:

```bash
git remote add origin https://github.com/<ВАШ_ЛОГИН>/amy-chat.git
git branch -M main
git push -u origin main
```

Альтернатива через GitHub CLI (нужен авторизованный `gh`):

```bash
gh auth login
gh repo create amy-chat --public --source=. --push
```

> ⚠️ Для push нужны ваши credentials (токен с правом `repo` или SSH-ключ).
> В текущем окружении токена GitHub нет, поэтому push нужно выполнить
> на своей машине либо предоставить токен (`GITHUB_TOKEN`).

## Откуда модель

«Мозг» Эми — модель из Hugging Face Space
**[pams90/Adult_Novel](https://huggingface.co/spaces/pams90/Adult_Novel/tree/main)**.
В том space используется `openai-community/gpt2` через
`transformers.pipeline("text-generation", ...)` с `do_sample=True,
temperature=0.8`. Наш сайт вызывает ровно ту же модель и параметры:

1. **gradio** — основной режим: запрос идёт в живой HF Space через
   `gradio_client` (`Client("pams90/Adult_Novel").predict(prompt, max_length)`);
2. **local** — fallback: если space недоступен, та же GPT-2 запускается
   локально через `transformers`;
3. **offline/guard** — если модели нет вовсе, отвечают каноничные реплики
   персонажа, чтобы диалог никогда не ломался.

## Как Эми «знает», кто она

GPT-2 — базовая модель без инструкционного тюнинга, поэтому личность
закрепляется тремя слоями (см. `amy_engine.py`):

* **системный промпт**: `You are Amy, a 23-year-old woman who works as a
  hotel manager…`;
* **few-shot «романный» диалог** User/Amy, где Эми уже называет своё имя,
  возраст и профессию;
* **страх личности (personality guard)**: ответ проверяется на связность и
  соответствие образу (на вопросы «как зовут?» нужен «Amy/Эми», «сколько
  лет?» — «23», «кем работаешь?» — упоминание отеля). Всё вне образа
  подменяется каноничной репликой.

Память диалога: последние 4 реплики добавляются в контекст промпта
(аналог «скользящего окна» из руководства по архитектуре).

## Запуск

```bash
pip install -r requirements.txt
python3 app.py            # http://localhost:7860
```

Переменные окружения (необязательно):

* `AMY_FORCE_MODE=gradio|local|offline` — принудительный выбор режима.

## Структура

```
app.py             Flask-сервер + REST API (/api/chat, /api/status, ...)
amy_engine.py      движок персонажа: подключение к GPT-2, промпты, guard
static/index.html  фронтенд чата (Character.ai-style)
requirements.txt
```

## API

| Метод | Путь                 | Описание                              |
|-------|----------------------|---------------------------------------|
| GET   | `/api/status`        | режим движка, системный промпт, модель |
| POST  | `/api/session/start` | создать диалог (приветствие Эми)       |
| POST  | `/api/chat`          | `{message, session_id}` → ответ Эми    |
| POST  | `/api/reset`         | очистить диалог                        |

## Что можно добавить дальше (по шагам из руководства)

* перенести на Next.js + streaming (SSE/WebSocket вместо полного ответа);
* Supabase/Firebase для пользователей и истории чатов;
* векторная память (саммари диалога) вместо скользящего окна;
* генерация аватара (DALL-E / Stable Diffusion), 👍/👎, озвучка (ElevenLabs);
* замена GPT-2 на более сильную open-source модель (Llama 3 / Mistral через
  Groq, Together) при тех же промптах — код это позволяет.

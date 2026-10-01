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

Основной «мозг» Эми — uncensored chat-модель с Hugging Face:
**[dphn/Dolphin3.0-Llama3.2-3B](https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B)**
(Llama 3.2 3B, дообученная на датасете Dolphin 3.0 без отказов по цензуре —
поддерживает «взрослые» разговоры и хорошо держит роль). Модель скачивается
и запускается локально через `transformers`.

Порядок выбора движка (авто-fallback):

1. **dolphin** — основной режим: локальная Dolphin 3.0; при нехватке RAM
   (<10 ГБ) автоматически пробуется int8-квантизация (`bitsandbytes`);
2. **gradio** — fallback: HF Space
   [pams90/Adult_Novel](https://huggingface.co/spaces/pams90/Adult_Novel/tree/main)
   (базовая `openai-community/gpt2`) через `gradio_client`;
3. **local** — та же GPT-2 локально через `transformers`
   (`do_sample=True, temperature=0.8`, как в исходном space);
4. **offline/guard** — если моделей нет вовсе, отвечают каноничные реплики
   персонажа, чтобы диалог никогда не ломался.

Переменные окружения: `AMY_MODEL` (id любой другой chat-модели на HF),
`AMY_FORCE_MODE=dolphin|gradio|local|offline` (принудительный режим).

Требования для основного режима: ~6.5 ГБ для весов (или ~4 ГБ в int8),
первый запуск скачивает модель из Hugging Face.

## Как Эми «знает», кто она

Личность закрепляется тремя слоями (см. `amy_engine.py`):

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

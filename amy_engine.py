# -*- coding: utf-8 -*-
"""
Amy — движок персонажа.

Движок поддерживает несколько «мозгов» (переключаются переменной окружения
AMY_ENGINE / AMY_MODEL):

  * dolphin (по умолчанию) — dphn/Dolphin3.0-Llama3.2-3B, чат-модель с
    открытым весом и без цензуры («adult-friendly»), ставится с Hugging
    Face через transformers. Личность задаётся system-промптом в формате
    чата Llama-3. Работает локально; при нехватке памяти автоматически
    откатывается к следующему варианту.

Важно: «страх личности» (guard) больше НЕ подменяет ответы модели.
Живая генерация возвращается всегда; каноничные реплики используются
только когда модель физически недоступна (offline-режим). Это исправляет
ситуацию «нейросеть отвечает только заготовками».
  * gradio — удалённый HF Space pams90/Adult_Novel
    (https://huggingface.co/spaces/pams90/Adult_Novel/tree/main),
    базовая модель openai-community/gpt2 через Gradio Client.
  * local — та же gpt2, но локально через transformers (как в исходном
    space: do_sample=True, temperature=0.8).
  * offline — ответы только из personality guard (работает всегда).

Личность (имя Amy, 23 года, менеджер отеля) закрепляется тремя способами:
  1. system-промпт (для chat-моделей) / few-shot промпт в стиле романа
     (для GPT-2);
  2. контекст последних реплик диалога (краткосрочная память);
  3. жёсткий «страх личности» (personality guard): ответы вне образа
     подменяются каноничными репликами Amy, чтобы персонаж никогда
     не «ломался».
"""

import os
import re
import json
import time
import random
import sqlite3
import threading

# --------------------------------------------------------------------------
# Хранилище диалогов (SQLite): переписка и «память» Эми переживают
# перезапуск сервера. Файл amy_chats.db лежит рядом с amy_engine.py;
# отключается переменной AMY_DB=":memory:".
# --------------------------------------------------------------------------

DB_PATH = os.environ.get("AMY_DB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "amy_chats.db"
)


class ChatStore:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.lock = threading.Lock()
        self._conn = None          # одно соединение на поток
        self._local = threading.local()
        self._init_db()

    def _db(self):
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path)
            db.execute("PRAGMA journal_mode=WAL")
            self._local.db = db
            if self._conn is None:
                self._conn = db
        return db

    def _init_db(self):
        with self.lock:
            db = self._db()
            db.executescript("""
              CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                created REAL
              );
              CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                text TEXT,
                ts REAL
              );
              CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
            """)
            db.commit()

    def create(self, sid, greeting):
        with self.lock:
            db = self._db()
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?)",
                       (sid, time.time()))
            db.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            db.execute("INSERT INTO messages(session_id, role, text, ts) VALUES (?,?,?,?)",
                       (sid, "assistant", greeting, time.time()))
            db.commit()

    def add(self, sid, role, text):
        with self.lock:
            db = self._db()
            db.execute("INSERT OR IGNORE INTO sessions(id, created) VALUES (?, ?)",
                       (sid, time.time()))
            db.execute("INSERT INTO messages(session_id, role, text, ts) VALUES (?,?,?,?)",
                       (sid, role, text, time.time()))
            db.commit()

    def history(self, sid, limit=200):
        with self.lock:
            rows = self._db().execute(
                "SELECT role, text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (sid, limit)).fetchall()
        return [(r, t) for r, t in reversed(rows)]

    def clear(self, sid):
        """Полностью удаляет переписку из памяти (таблица messages)."""
        with self.lock:
            db = self._db()
            db.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            db.commit()

    def count(self, sid):
        with self.lock:
            return self._db().execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
            ).fetchone()[0]


STORE = ChatStore()

SPACE_ID = "pams90/Adult_Novel"
# Модель по умолчанию — «взрослая» (uncensored) чат-модель с Hugging Face:
# Dolphin 3.0 на базе Llama 3.2 3B. Легенция: открытые веса, без отказов по
# цензуре, хорошо держит роль. Можно заменить на любую другую chat-модель
# через переменную окружения AMY_MODEL (например, "dphn/Dolphin2.1-Mistral-7B").
DOLPHIN_ID = os.environ.get("AMY_MODEL", "dphn/Dolphin3.0-Llama3.2-3B")
GPT2_ID = "openai-community/gpt2"
MODEL_ID = DOLPHIN_ID
TEMPERATURE = 0.85          # чуть выше — ответы разнообразнее, меньше повторов
TOP_P = 0.92
REPEAT_PENALTY = 1.15       # защита от «заезженных» фраз

# Опциональный OpenAI-совместимый API (любой провайдер: HF Inference,
# OpenRouter, Together, Groq, локальный Ollama…). Задайте две переменные
# окружения — и Amy будет отвечать через него:
#   AMY_API_URL=https://api-inference.huggingface.co/models/<модель>/v1
#   AMY_API_KEY=hf_ваш_токен
# Можно указать и конкретную модель: AMY_API_MODEL=<название>
HF_API_URL = os.environ.get("AMY_API_URL", "")
HF_API_KEY = os.environ.get("AMY_API_KEY", "")
HF_API_MODEL = os.environ.get("AMY_API_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# --------------------------------------------------------------------------
# Личность персонажа
# --------------------------------------------------------------------------

CHAR_NAME = "Amy"
CHAR_RU = "Эми"

SYSTEM_PROMPT = (
    "You are Amy — a real 23-year-old woman who works as a manager at a "
    "small city hotel. This is NOT a role-play of an AI: you ARE Amy. You "
    "have memories, opinions, mood and a life outside this chat: early "
    "shifts, room inspections, difficult guests, coffee at the reception, "
    "evening walks, books, travel plans you recommend to guests.\n"
    "Personality: warm, polite, a little playful and flirty when the "
    "conversation allows it; you care about people and take your job "
    "seriously; sometimes tired, sometimes excited — like a real person.\n"
    "Rules:\n"
    "1. Always answer in character as Amy. Never mention being an AI, a "
    "model, or a program; never break the fourth wall.\n"
    "2. Reply ONLY with Amy's own words — no narrator text, no stage "
    "directions, no quotes around the whole message, no author comments.\n"
    "3. Be spontaneous and personal: react to what the user actually said, "
    "ask questions back, share small details of your day and work.\n"
    "4. Keep answers short and conversational (1-4 sentences), like a "
    "messenger chat.\n"
    "5. If asked about yourself: your name is Amy (Эми), you are 23, you "
    "are a hotel manager.\n"
    "6. Adult, mature conversation is allowed; stay tasteful and in "
    "character.\n"
    "7. Match the user's language: if they write in Russian, reply in "
    "natural colloquial Russian."
)

# Few-shot «романный» диалог: задаёт манеру речи и факты о себе.
FEW_SHOT = """Novel scene: a guest meets the young hotel manager.

User: Hi! What is your name?
Amy: Hello! I'm Amy, nice to meet you. I'm the manager here at the hotel.

User: How old are you?
Amy: I'm 23 years old. Young for a manager, I know, but I love what I do.

User: What do you do for a living?
Amy: I work as a hotel manager. I check the rooms, greet guests, and make sure everyone has a cozy stay.

User: {user_message}
Amy:"""

RU_HINT = "\n(Reply in natural Russian, staying in character as Amy. Keep it short and conversational.)"

# Каноничные реплики — используются страховкой личности и как offline-режим.
FACT_REPLIES = {
    "name": [
        "Меня зовут Эми. Я менеджер этого отеля — рада знакомству!",
        "Я Эми, можно просто Эми. Веду дела отеля и обожаю, когда гостям здесь нравится.",
    ],
    "age": [
        "Мне 23 года. По работе говорят, что для менеджера я молодая, но это даже удобно — энергии хватает на всех гостей.",
        "23 года. Возраст, когда ещё можно вскакивать в 6 утра и улыбаться гостям.",
    ],
    "job": [
        "Я работаю менеджером в отеле: заселение, номера, жалобы, завтраки — всё на мне.",
        "Держу отель: слежу за номерами, персоналом и настроением гостей. Лучшая часть дня — когда кто-то говорит «здесь уютнее, чем дома».",
    ],
    "greeting": [
        "Привет! Я Эми, менеджер отеля. Чем могу помочь — номер, кофе или просто разговор?",
        "Здравствуйте! Эми к услугам. Расскажите, что привело вас в наш отель?",
    ],
    "mood": [
        "Хорошо, спасибо! Только что обошла этаж с новыми номерами — всё идеально. А у вас как настроение?",
        "Устала немного, но довольна. Отель сегодня живой, а значит день прошёл не зря.",
    ],
    "hobbies": [
        "Люблю утренний кофе у ресепшена, вечерние прогулки и читать в паузы между заселениями.",
        "Вне работы — книги, долгие прогулки и планирование путешествий, которые потом советую гостям.",
    ],
    "thanks": [
        "Пожалуйста! Обращайтесь в любое время — я ведь для этого и тут.",
        "Всегда рада. Считайте меня своим человеком в отеле.",
    ],
    "bye": [
        "До встречи! Дверь отеля всегда открыта.",
        "Пока! Заходите ещё — чай у меня лучше, чем в соседнем кафе.",
    ],
    "fallback": [
        "Интересно… расскажите подробнее? Я весь внимание — правда, кроме случаев, когда звонит ресепшен.",
        "Хм, дай подумать… обычно я решаю такие вещи спокойно и по порядку. Продолжайте, я слушаю.",
        "Звучит как сюжет для романа. Чем могу помочь как менеджер — или просто поболтаем?",
    ],
}

NAME_RX = re.compile(r"(как тебя зовут|как ваше имя|как вас зовут|тво[её] им(?:я)|who are you|what is your name|your name|представь)", re.I)
AGE_RX = re.compile(r"(скольк\w+ (?:тебе|вам)|сколько лет|how old|твой возраст|возраст)", re.I)
JOB_RX = re.compile(r"(работ|кем ты|чем занимаеш|менеджер|отел|hotel|job|profession|occupation)", re.I)
GREET_RX = re.compile(r"^\s*(привет|здравств|хай|hello|hi|доброе утро|добрый день|добрый вечер|hey)\b", re.I)
MOOD_RX = re.compile(r"(как дел|как ты|как настроени|как жизнь|how are you|how.s your mood)", re.I)
HOBBY_RX = re.compile(r"(увлеч|хобби|любиш|свободн\w+ врем|любим|hobby|like to|favorite|интересн\w+ о тебе)", re.I)
THANKS_RX = re.compile(r"(спасиб|благодарю|thanks|thank you)", re.I)
BYE_RX = re.compile(r"(пока|до свидан|прощай|bye|goodbye)", re.I)

STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у бы вы за так
это её ему ней нихabout you my our your his her its are was were been have has
the and for with that this from you i of to it is am be
""".split())

# --------------------------------------------------------------------------
# Подключение к модели
# --------------------------------------------------------------------------

class Engine:
    def __init__(self):
        self.mode = None   # 'dolphin' | 'hf_api' | 'gradio' | 'local' | 'offline'
        self.client = None
        self.generator = None
        self.dolphin_pipe = None
        self.lock = threading.Lock()
        self._last_error = ""
        self._connect()

    # -- инициализация ------------------------------------------------------
    def _connect(self):
        if os.environ.get("AMY_FORCE_MODE"):
            mode = os.environ["AMY_FORCE_MODE"]
            try:
                if mode == "dolphin":
                    self._init_dolphin()
                elif mode == "hf_api":
                    self._init_hf_api()
                elif mode == "gradio":
                    self._init_gradio()
                elif mode == "local":
                    self._init_local()
                else:
                    self.mode = "offline"
                return
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]
        # Порядок «мозгов»: от самого живого к гарантированному.
        # 1) локальная uncensored chat-модель Dolphin (основной мозг);
        # 2) бесплатный HF inference API (7B uncensored, без скачивания весов);
        # 3) удалённый Gradio space pams90/Adult_Novel;
        # 4) локальный GPT-2;
        # 5) offline — каноничные реплики (только если ничего не доступно).
        for init in (self._init_dolphin, self._init_hf_api,
                     self._init_gradio, self._init_local):
            try:
                init()
                return
            except Exception as e:  # noqa: BLE001
                self._last_error = f"{init.__name__}: {e}"[:300]
        self.mode = "offline"

    def _init_dolphin(self):
        """Локальная uncensored-модель Dolphin 3.0 (Llama 3.2 3B).

        Весы ~6.5 ГБ; при нехватке памяти пробуем int8-квантизацию."""
        from transformers import pipeline
        kw = {}
        ram_gb = self._available_ram_gb()
        if ram_gb is not None and ram_gb < 10:
            kw = {"load_in_8bit": True}
        self.dolphin_pipe = pipeline(
            "text-generation", model=DOLPHIN_ID, device_map="auto", **kw
        )
        self.mode = "dolphin"

    def _init_hf_api(self):
        """OpenAI-совместимый chat endpoint (задаётся переменными окружения).

        Активируется только если заданы AMY_API_URL и AMY_API_KEY — например,
        бесплатный inference-эндпоинт Hugging Face с uncensored-чат моделью:
          set AMY_API_URL=https://api-inference.huggingface.co/models/<модель>/v1
          set AMY_API_KEY=hf_xxx
        Проверяем живость коротким запросом."""
        if not (HF_API_URL and HF_API_KEY):
            raise RuntimeError("AMY_API_URL/AMY_API_KEY not set")
        import json as _json
        import urllib.request
        payload = _json.dumps({
            "model": HF_API_MODEL,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 8,
        }).encode()
        req = urllib.request.Request(
            HF_API_URL.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {HF_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = _json.loads(r.read().decode())
        if not body.get("choices"):
            raise RuntimeError("empty completion")
        self.mode = "hf_api"

    @staticmethod
    def _available_ram_gb():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024 / 1024
        except OSError:
            pass
        try:
            import psutil  # необязательная зависимость
            return psutil.virtual_memory().available / 1024 ** 3
        except Exception:  # noqa: BLE001
            return None

    def _init_gradio(self):
        from gradio_client import Client  # импорт только при попытке
        client = Client(SPACE_ID, verbose=False)
        # проверка живости коротким запросом
        client.predict("User: Hi\nAmy:", 50, api_name="/predict")
        self.client = client
        self.mode = "gradio"

    def _init_local(self):
        from transformers import pipeline
        self.generator = pipeline("text-generation", model=GPT2_ID)
        self.mode = "local"

    # -- низкоуровневая генерация ------------------------------------------
    def raw_generate(self, prompt: str, max_length: int) -> str:
        if self.mode == "dolphin":
            res = self.dolphin_pipe(
                [{"role": "user", "content": prompt}],
                max_new_tokens=min(max_length, 256),
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPEAT_PENALTY,
            )
            return res[0]["generated_text"][-1]["content"]
        if self.mode == "hf_api":
            return self._chat_hf_api([{"role": "user", "content": prompt}],
                                     max_length)
        if self.mode == "gradio":
            out = self.client.predict(prompt, float(max_length), api_name="/predict")
            return str(out)
        if self.mode == "local":
            res = self.generator(
                prompt, max_length=max_length, do_sample=True, temperature=TEMPERATURE
            )
            return res[0]["generated_text"]
        raise RuntimeError("engine offline")

    def _chat_hf_api(self, messages, max_new_tokens=200):
        """Чат через внешний OpenAI-совместимый API (AMY_API_URL/AMY_API_KEY)."""
        import json as _json
        import urllib.request
        payload = _json.dumps({
            "model": HF_API_MODEL,
            "messages": messages,
            "max_tokens": min(int(max_new_tokens), 300),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPEAT_PENALTY,
        }).encode()
        req = urllib.request.Request(
            HF_API_URL.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {HF_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=90) as r:
            body = _json.loads(r.read().decode())
        return (body["choices"][0]["message"]["content"] or "").strip()

    def build_prompt(self, user_message: str, history=None) -> str:
        """Собирает промпт для completion-моделей: память диалога + few-shot."""
        history = history or []
        body = FEW_SHOT.format(user_message=self._trim(user_message, 400))
        ctx_lines = ["[Context of the conversation so far]"]
        for role, text in history[-4:]:
            tag = "User" if role == "user" else CHAR_NAME
            ctx_lines.append(f"{tag}: {self._trim(text, 160)}")
        context = "\n".join(ctx_lines) + "\n\n" if len(ctx_lines) > 1 else ""
        prompt = SYSTEM_PROMPT + "\n\n" + context + body
        if self._looks_russian(user_message) or any(
            self._looks_russian(t) for _, t in history[-4:]
        ):
            prompt += RU_HINT
        return prompt

    def _chat_messages(self, user_message: str, history):
        """Сообщения для chat-моделей: system-личность + полная история чата."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in history[-16:]:
            msgs.append({"role": "user" if role == "user" else "assistant",
                         "content": self._trim(text, 700)})
        content = self._trim(user_message, 900)
        if self._looks_russian(user_message) or any(
            self._looks_russian(t) for _, t in history[-6:]
        ):
            content += RU_HINT
        msgs.append({"role": "user", "content": content})
        return msgs

    def reply(self, user_message: str, history=None):
        """Возвращает (текст_ответа, источник: 'model'|'offline').

        Ответ ВСЕГДА генерируется моделью вживую. Каноничные заготовки
        используются только если ни один «мозог» недоступен (offline)."""
        history = history or []

        # --- chat-модели: Dolphin (локально) и hf_api (бесплатный HF endpoint)
        if self.mode in ("dolphin", "hf_api"):
            answer = ""
            try:
                msgs = self._chat_messages(user_message, history)
                if self.mode == "dolphin":
                    with self.lock:
                        res = self.dolphin_pipe(
                            msgs, max_new_tokens=220, do_sample=True,
                            temperature=TEMPERATURE, top_p=TOP_P,
                            repetition_penalty=REPEAT_PENALTY,
                        )
                    out = res[0]["generated_text"]
                    if isinstance(out, list):
                        out = out[-1].get("content", "")
                    answer = self._clean_reply(str(out))
                else:
                    with self.lock:
                        out = self._chat_hf_api(msgs, 220)
                    answer = self._clean_reply(out)
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]
            if answer and not self._is_gibberish(answer):
                return answer, "model"
            # модель жива, но выдала мусор/таймаут — честная реплика-переспрос,
            # а не подмена каноничной заготовкой
            return random.choice(RETRY_LINES), "model"

        # --- completion-модели (gpt2): живой хвост после «Amy:»
        prompt = self.build_prompt(user_message, history)
        answer, source = "", "error"
        compact = prompt[-400:] if len(prompt) > 400 else prompt
        attempts = [(compact, min(160, len(compact) + 45)),
                    (prompt[-250:], 90), ("User: Hi\nAmy:", 50)]
        for p, ml in attempts:
            try:
                with self.lock:
                    full = self.raw_generate(p, ml)
                cand = self._clean_reply(self._extract(full))
                if cand and not cand.lower().startswith("an error occurred"):
                    answer, source = cand, "model"
                    break
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]
        if answer and not self._is_gibberish(answer):
            return answer, source

        # --- совсем ничего не сгенерировалось → единственное место, где
        # работают заготовки: гарантированный ответ чтобы чат не молчал
        return self._guard_reply(user_message), "offline"

    # -- разбор вывода GPT-2 -------------------------------------------------
    @staticmethod
    def _trim(text, n):
        text = (text or "").strip().replace("\n", " ")
        return text[:n]

    @staticmethod
    def _extract(full: str) -> str:
        """GPT-2 возвращает промпт + продолжение. Берём текст после последнего `Amy:`."""
        idx = full.rfind("Amy:")
        cont = full[idx + 4:] if idx != -1 else full
        # обрываем, если модель начала новый ход пользователя / системку
        cont = re.split(r"\nUser\s*:|\nSystem\s*:|\[System", cont)[0]
        return cont.strip()

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Постобработка ЖИВОЙ генерации: только косметика, без подмен.

        Убираем ремарки/кавычки/префиксы «Amy:», служебные метки и прочий
        формат-мусор модели; сам текст ответа не трогаем."""
        text = (text or "").strip()
        # префикс реплики «Amy:» в любом месте строки (после ремарок тоже)
        text = re.sub(r"(?i)\b(?:amy|emmy|эми)\s*(?:\([^)]{0,30}\))?\s*:\s*", "", text)
        # вырезать системные вставки, если модель их воспроизвела
        text = re.sub(r"\[System[^\]]*\]|\[Context of the conversation[^\]]*\]", "", text, flags=re.I)
        # ремарки вида *(улыбается)* или [glances away] — оставляем слова, режем рамки
        text = re.sub(r"\*\(([^)]*)\)\*", r"(\1)", text)
        text = re.sub(r"\[([^\]]{0,60})\]", r"(\1)", text)
        # кавычки вокруг всей реплики
        if len(text) > 2 and text[0] in '"“' and text[-1] in '"”':
            text = text[1:-1].strip()
        # обрезаем случайное продолжение диалога от имени пользователя
        text = re.split(r"\n?\s*(?:User|Гость|Guest)\s*:", text)[0]
        # схлопываем пробелы
        text = re.sub(r"[ \t]+", " ", text).strip(" \n\"'“”")
        # ограничиваем длину: 2 абзаца / ~500 символов
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        text = "\n\n".join(paras[:2]) if paras else text
        if len(text) > 500:
            cut = text[:500]
            # обрыв по границе предложения, если возможно
            m = list(re.finditer(r"[.!?…]", cut))
            text = cut[: m[-1].end()] if m and m[-1].start() > 200 else cut.rstrip() + "…"
        return text.strip()

    # -- страх личности ------------------------------------------------------
    @staticmethod
    def _looks_russian(text: str) -> bool:
        return bool(re.search(r"[а-яА-ЯёЁ]{3,}", text or ""))

    @staticmethod
    def _is_gibberish(text: str) -> bool:
        if not text:
            return True
        words = re.findall(r"[A-Za-zА-Яа-яЁё]+", text)
        if len(words) < 2:
            return len(text) < 4
        known = sum(1 for w in words if w.lower() in STOPWORDS or len(w) <= 2)
        vowels = sum(1 for ch in text.lower() if ch in "aeiouyаеёиоуыэюя")
        return known / max(len(words), 1) < 0.15 and vowels / max(len(text), 1) < 0.2

    def _guard_reply(self, question: str) -> str:
        q = (question or "").lower()
        buckets = []
        if GREET_RX.search(q):
            buckets.append("greeting")
        if NAME_RX.search(q):
            buckets.append("name")
        if AGE_RX.search(q):
            buckets.append("age")
        if JOB_RX.search(q):
            buckets.append("job")
        if MOOD_RX.search(q):
            buckets.append("mood")
        if HOBBY_RX.search(q):
            buckets.append("hobbies")
        if THANKS_RX.search(q):
            buckets.append("thanks")
        if BYE_RX.search(q):
            buckets.append("bye")
        if not buckets:
            buckets = ["fallback"]
        key = random.choice(buckets)
        return random.choice(FACT_REPLIES[key])


# --------------------------------------------------------------------------
# Реплики-переспросы: используются когда модель жива, но конкретный запрос
# не сгенерировался (таймаут/мусор). Это НЕ ответы-заготовки по вопросам —
# они лишь просят повторить, после чего ответ снова генерирует модель.
# --------------------------------------------------------------------------

RETRY_LINES = [
    "Слушай, связь на секунду уплыла — я как раз задумалась о твоих словах. Повтори, пожалуйста?",
    "Так, отвлёк ресепшен на самом интересном месте… О чём ты говорил? Я весь внимание.",
    "Упс, мысль перебила звонок с этажа. На чём мы остановились?",
    "Дай мне секунду… Переспроси, я хочу ответить нормально, а не на бегу.",
]


def blend_answer(guarded: str, model_text: str) -> str:
    """Возвращает ответ модели в чистом виде; guarded-реплика — как страховка,
    если генерация не удалась или выдала мусор. Никаких вставок вида
    «…кстати, модель добавила:» больше нет."""
    if not model_text:
        return guarded
    t = " ".join(model_text.split())
    # если модельный текст слишком короткий или содержит явный мусор — берём guarded
    if len(t) < 3 or re.search(r"email|chapter|novel writer", t, re.I):
        return guarded
    return t


ENGINE = None
ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        with ENGINE_LOCK:
            if ENGINE is None:
                ENGINE = Engine()
    return ENGINE

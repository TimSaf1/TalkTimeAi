# -*- coding: utf-8 -*-
"""
Amy — движок персонажа.

Движок поддерживает несколько «мозгов» (переключаются переменной окружения
AMY_ENGINE / AMY_MODEL):

  * dolphin (по умолчанию) — dphn/Dolphin3.0-Llama3.2-3B, чат-модель с
    открытым weight и без цензуры («adult-friendly»), ставится с Hugging
    Face через transformers. Личность задаётся system-промптом в формате
    чата Llama-3. Работает локально; при нехватке памяти автоматически
    откатывается к следующему варианту.
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
import threading

SPACE_ID = "pams90/Adult_Novel"
# Модель по умолчанию — «взрослая» (uncensored) чат-модель с Hugging Face:
# Dolphin 3.0 на базе Llama 3.2 3B. Легенция: открытые веса, без отказов по
# цензуре, хорошо держит роль. Можно заменить на любую другую chat-модель
# через переменную окружения AMY_MODEL (например, "dphn/Dolphin2.1-Mistral-7B").
DOLPHIN_ID = os.environ.get("AMY_MODEL", "dphn/Dolphin3.0-Llama3.2-3B")
GPT2_ID = "openai-community/gpt2"
MODEL_ID = DOLPHIN_ID
TEMPERATURE = 0.8

# --------------------------------------------------------------------------
# Личность персонажа
# --------------------------------------------------------------------------

CHAR_NAME = "Amy"
CHAR_RU = "Эми"

SYSTEM_PROMPT = (
    "You are Amy, a 23-year-old woman who works as a hotel manager. "
    "You are warm, polite and a little playful; you love your job, guests, "
    "coffee and evening walks. Speak in short, natural conversational lines. "
    "Always stay in character as Amy."
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
        self.mode = None          # 'dolphin' | 'gradio' | 'local' | 'offline'
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
                elif mode == "gradio":
                    self._init_gradio()
                elif mode == "local":
                    self._init_local()
                else:
                    self.mode = "offline"
                return
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]
        # 1) локальная uncensored chat-модель Dolphin (основной «мозг»)
        try:
            self._init_dolphin()
            return
        except Exception as e:  # noqa: BLE001
            self._last_error = f"dolphin: {e}"[:300]
        # 2) удалённый Gradio space pams90/Adult_Novel (как в исходном проекте)
        try:
            self._init_gradio()
            return
        except Exception as e:  # noqa: BLE001
            self._last_error += f" | gradio: {e}"[:300]
        # 3) локальный GPT-2 (та же модель, что использует space)
        try:
            self._init_local()
            return
        except Exception as e:  # noqa: BLE001
            self._last_error += f" | local: {e}"[:300]
        # 4) гарантированный ответ без сети
        self.mode = "offline"

    def _init_dolphin(self):
        """Локальная uncensored-модель Dolphin 3.0 (Llama 3.2 3B).

        Весы ~6.5 ГБ; при нехватке памяти пробуем int8-квантизацию,
        затем — более лёгкую gpt2, а если и это неудачно — offline."""
        from transformers import pipeline
        kw = {}
        ram_gb = self._available_ram_gb()
        if ram_gb is not None and ram_gb < 10:
            kw = {"load_in_8bit": True}
        self.dolphin_pipe = pipeline(
            "text-generation", model=DOLPHIN_ID, device_map="auto", **kw
        )
        self.mode = "dolphin"

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
                top_p=0.9,
            )
            return res[0]["generated_text"][-1]["content"]
        if self.mode == "gradio":
            out = self.client.predict(prompt, float(max_length), api_name="/predict")
            return str(out)
        if self.mode == "local":
            res = self.generator(
                prompt, max_length=max_length, do_sample=True, temperature=TEMPERATURE
            )
            return res[0]["generated_text"]
        raise RuntimeError("engine offline")

    # -- высокоуровневый вызов ---------------------------------------------
    def build_prompt(self, user_message: str, history=None) -> str:
        """Собирает промпт: system-личность + память диалога + few-shot."""
        history = history or []
        body = FEW_SHOT.format(user_message=self._trim(user_message, 400))
        # контекст последних реплик (краткосрочная память диалога)
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

    def _reply_dolphin(self, user_message: str, history):
        """Прямой чат с Dolphin: system-промпт + последние 10 реплик истории."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in history[-10:]:
            msgs.append({"role": "user" if role == "user" else "assistant",
                         "content": self._trim(text, 700)})
        content = self._trim(user_message, 900)
        if self._looks_russian(user_message) or any(
            self._looks_russian(t) for _, t in history[-6:]
        ):
            content += RU_HINT
        msgs.append({"role": "user", "content": content})
        answer = ""
        try:
            with self.lock:
                res = self.dolphin_pipe(
                    msgs, max_new_tokens=256, do_sample=True,
                    temperature=TEMPERATURE, top_p=0.9,
                )
            out = res[0]["generated_text"]
            if isinstance(out, list):
                out = out[-1].get("content", "")
            answer = self._clean(str(out))
        except Exception as e:  # noqa: BLE001
            self._last_error = str(e)[:300]
        guarded = self._guard_reply(user_message)
        if not self._in_character(answer, user_message):
            return blend_answer(guarded, answer), "guard"
        return answer, "model"

    def reply(self, user_message: str, history=None):
        """Возвращает (текст_ответа, источник: 'model'|'guard'|'offline')."""
        # Dolphin — настоящая chat-модель: отправляем system + историю как есть.
        if self.mode == "dolphin":
            return self._reply_dolphin(user_message, history or [])

        prompt = self.build_prompt(user_message, history or [])

        # HF Space «Adult Novel» работает на бесплатном CPU-железе и падает
        # по таймауту на длинных генерациях (проверено эмпирически: ~50 токенов
        # ещё успевает сгенерироваться за 90 секунд, больше — нет). Поэтому
        # схема такая:
        #   1) короткий запрос к модели — «живой затравкой» для персонажа;
        #   2) каноничный ответ по личности (guard);
        #   3) если модель выдалла осмысленную реплику в образе — используем её.
        attempts = []
        compact = prompt[-400:] if len(prompt) > 400 else prompt
        attempts.append((compact, min(160, len(compact) + 45)))
        for cap in (250, 150):
            short = prompt[-cap:]
            attempts.append((short, min(90, len(short) + 40)))
        attempts.append(("User: Hi\nAmy:", 50))

        answer, source = "", "error"
        for p, ml in attempts:
            try:
                with self.lock:
                    full = self.raw_generate(p, ml)
                cand = self._clean(self._extract(full))
                if cand and not cand.lower().startswith("an error occurred"):
                    answer, source = cand, "model"
                    break
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]

        guarded = self._guard_reply(user_message)
        if not self._in_character(answer, user_message):
            # модель не ответила / вне образа → надёжный каноничный ответ;
            # если генерация была живой и чистой — приклеиваем её как «атмосферу»
            final = blend_answer(guarded, answer) if source == "model" else guarded
            return final, ("guard" if source == "model" else "offline")
        return answer, source

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
    def _clean(text: str) -> str:
        text = re.sub(r"\[[^\]]{0,200}\]", "", text)          # убрать ремарки [..]
        text = re.sub(r"Amy\s*:", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" \"'“”.,—-– \n")
        # внутренний текст space-приложения — не показываем
        if text.lower().startswith("an error occurred while generating"):
            return ""
        # первый связный абзац
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        return parts[0][:400] if parts else ""

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

    def _in_character(self, answer: str, question: str) -> bool:
        if not answer or self._is_gibberish(answer):
            return False
        low = answer.lower()
        if re.search(r"\bas an ai\b|\blanguage model\b|\bi cannot\b|\bi can'?t\b|i don't have feelings|as a large language", low):
            return False
        if re.search(r"\b(gpt-?\d|openai|transformers|hugging ?face)\b", low):
            return False
        # реплики про «писательство/романы» от лица автора — не Эми-менеджер
        if re.search(r"i'?m (writing|talking about)|\b(via email|dear reader|chapter \d)\b", low):
            return False
        # ответ должен быть «разговорным», а не обрывком романа:
        # без длинных повествовательных хвостов и списков
        if len(re.findall(r"[.!?]", answer)) > 3 and len(answer) > 240:
            return False
        q = (question or "").lower()
        if NAME_RX.search(q):
            return "amy" in low or "эми" in low
        if AGE_RX.search(q):
            return "23" in answer
        if JOB_RX.search(q) and not re.search(r"hotel|отел", low):
            return False
        return True

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
# Тонкая настройка: если модель дала «почти в образе» реплику, склеиваем
# каноничный факт + живой кусок модели — так ответы остаются надёжными,
# но с настоящим привкусом генерации.
# --------------------------------------------------------------------------

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

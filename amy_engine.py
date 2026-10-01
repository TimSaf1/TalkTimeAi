# -*- coding: utf-8 -*-
"""
Amy — движок персонажа.

Модель-«мозг» берётся из Hugging Face Space pams90/Adult_Novel:
https://huggingface.co/spaces/pams90/Adult_Novel/tree/main

Тот space работает на базовой модели openai-community/gpt2 через
`transformers.pipeline("text-generation", ...)` (см. app.py в space) с
параметрами do_sample=True, temperature=0.8.  Мы используем ровно ту же
модель и те же параметры — либо удалённо через Gradio API space
(gradio_client), либо локально через transformers, если сети нет.

GPT-2 — маленькая модель без инструкционного тюнинга, поэтому личность
(имя Amy, 23 года, менеджер отеля) закрепляется двумя способами:
  1. few-shot промпт в стиле романа (диалог User/Amy) — то, что GPT-2 умеет;
  2. жёсткий «страх личности» (personality guard): ответы вне образа
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
MODEL_ID = "openai-community/gpt2"
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

RU_HINT = "\n(Amy answers in Russian, in character.)"

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
        self.mode = None          # 'gradio' | 'local' | 'offline'
        self.client = None
        self.generator = None
        self.lock = threading.Lock()
        self._last_error = ""
        self._connect()

    # -- инициализация ------------------------------------------------------
    def _connect(self):
        if os.environ.get("AMY_FORCE_MODE"):
            mode = os.environ["AMY_FORCE_MODE"]
            try:
                if mode == "gradio":
                    self._init_gradio()
                elif mode == "local":
                    self._init_local()
                else:
                    self.mode = "offline"
                return
            except Exception as e:  # noqa: BLE001
                self._last_error = str(e)[:300]
        # 1) удалённый Gradio space (как в исходном проекте)
        try:
            self._init_gradio()
            return
        except Exception as e:  # noqa: BLE001
            self._last_error = f"gradio: {e}"[:300]
        # 2) локальный GPT-2 (та же модель, что использует space)
        try:
            self._init_local()
            return
        except Exception as e:  # noqa: BLE001
            self._last_error += f" | local: {e}"[:300]
        # 3) гарантированный ответ без сети
        self.mode = "offline"

    def _init_gradio(self):
        from gradio_client import Client  # импорт только при попытке
        client = Client(SPACE_ID, verbose=False)
        # проверка живости коротким запросом
        client.predict("User: Hi\nAmy:", 50, api_name="/predict")
        self.client = client
        self.mode = "gradio"

    def _init_local(self):
        from transformers import pipeline
        self.generator = pipeline("text-generation", model=MODEL_ID)
        self.mode = "local"

    # -- низкоуровневая генерация ------------------------------------------
    def raw_generate(self, prompt: str, max_length: int) -> str:
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

    def reply(self, user_message: str, history=None):
        """Возвращает (текст_ответа, источник: 'model'|'guard'|'offline')."""
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
    if not model_text:
        return guarded
    t = model_text.strip()
    # берём только первую фразу модельного хвоста, если она короткая и чистая
    first = re.split(r"(?<=[.!?]) ", t)[0]
    if 8 <= len(first) <= 90 and not re.search(r"email|chapter|novel|writer", first, re.I):
        return f"{guarded} …кстати, модель добавила: «{first}»"
    return guarded


ENGINE = None
ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        with ENGINE_LOCK:
            if ENGINE is None:
                ENGINE = Engine()
    return ENGINE

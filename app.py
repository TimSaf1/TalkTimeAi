# -*- coding: utf-8 -*-
"""
Amy — пробный сайт-переписка с нейросетью в стиле Character.ai.

«Мозг» персонажа — модель из Hugging Face Space pams90/Adult_Novel
(openai-community/gpt2, text-generation), подключается через Gradio API.
Личность (Эми, 23 года, менеджер отеля) задаётся системным промптом,
few-shot диалогом и «страхом личности».

Запуск:  python3 app.py   ->  http://localhost:7860
"""

import os
import sys
import time
import uuid

# Запуск из любой папки: ищем amy_engine.py рядом со скриптом,
# в текущей директории или в ./src — и добавляем найденную папку в sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_engine_dir():
    for d in (_HERE, os.getcwd(), os.path.join(_HERE, "src")):
        if os.path.isfile(os.path.join(d, "amy_engine.py")):
            return d
    return None

_ENGINE_DIR = _find_engine_dir()
if _ENGINE_DIR and _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

try:
    from amy_engine import SYSTEM_PROMPT, DOLPHIN_ID, STORE, get_engine
except ModuleNotFoundError:
    sys.exit(
        "ОШИБКА: не найден файл amy_engine.py!\n"
        "Он должен лежать в одной папке с app.py.\n"
        f"Искали в: {_HERE} | {os.getcwd()}\n"
        "Решение: скачайте ВСЕ файлы репозитория (Code -> Download ZIP)\n"
        "или клонируйте:  git clone https://github.com/TimSaf1/TalkTimeAi.git"
    )

from flask import Flask, jsonify, request, send_from_directory

# Папка static ищетcя рядом с app.py (или в cwd), а не относительно места запуска,
# иначе при запуске из другой папки GET / вернёт 404.
_STATIC_DIR = next(
    (d for d in (os.path.join(_HERE, "static"), os.path.join(os.getcwd(), "static"))
     if os.path.isdir(d)),
    os.path.join(_HERE, "static"),
)
app = Flask(__name__, static_folder=_STATIC_DIR, static_url_path="/static")

GREETING = (
    "Привет! Я Эми — мне 23 года, и я менеджер этого отеля. "
    "Устроимся поудобнее? Расскажи, как прошёл твой день."
)

sessions = {}  # id -> {"messages": [(role, text)], "created": ts}  (кэш поверх SQLite)


def _session(sid):
    """Возвращает кэшированную сессию; при промахе загружает историю из БД."""
    s = sessions.get(sid)
    if s is None:
        hist = STORE.history(sid)
        if hist:
            s = {"messages": hist, "created": time.time()}
            sessions[sid] = s
    return s


@app.after_request
def add_headers(resp):
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.get("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.get("/favicon.ico")
def favicon():
    # тихая заглушка, чтобы не спамить 404 в логах
    return "", 204


@app.get("/api/status")
def status():
    eng = get_engine()
    return jsonify(
        {
            "character": {"name": "Amy", "name_ru": "Эми", "age": 23,
                          "job": "менеджер отеля"},
            "model_source": "https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B",
            "base_model": DOLPHIN_ID,
            "fallback_space": "https://huggingface.co/spaces/pams90/Adult_Novel (gpt2)",
            "engine_mode": eng.mode,          # dolphin | hf_api | gradio | local | offline
            "external_api": bool(os.environ.get("AMY_API_URL")),
            "system_prompt": SYSTEM_PROMPT,
            "last_error": eng._last_error,
            "storage": "sqlite:" + os.path.basename(STORE.path),
        }
    )


@app.post("/api/session/start")
def start_session():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    # если клиент прислал старый id и в БД есть переписка — продолжаем её
    if sid and STORE.count(sid):
        s = _session(sid)
        return jsonify({"session_id": sid, "resumed": True,
                        "messages": [{"role": r, "text": t} for r, t in s["messages"]]})
    sid = uuid.uuid4().hex[:12]
    STORE.create(sid, GREETING)
    sessions[sid] = {"messages": [("assistant", GREETING)], "created": time.time()}
    return jsonify({"session_id": sid, "resumed": False,
                    "messages": [{"role": "assistant", "text": GREETING}]})


@app.get("/api/history/<sid>")
def history(sid):
    s = _session(sid)
    if not s:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"messages": [{"role": r, "text": t} for r, t in s["messages"]]})


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    sid = data.get("session_id")
    if not message:
        return jsonify({"error": "empty message"}), 400
    s = _session(sid) if sid else None
    if s is None:
        sid = uuid.uuid4().hex[:12]
        STORE.create(sid, GREETING)
        s = {"messages": [("assistant", GREETING)], "created": time.time()}
        sessions[sid] = s

    history = list(s["messages"])
    eng = get_engine()
    t0 = time.time()
    answer, source = eng.reply(message, history)
    s["messages"].append(("user", message))
    s["messages"].append(("assistant", answer))
    STORE.add(sid, "user", message)
    STORE.add(sid, "assistant", answer)
    # история чата ограничена со стороны модели, но память (БД) полная
    if len(s["messages"]) > 60:
        s["messages"] = s["messages"][-60:]
    return jsonify({
        "session_id": sid,
        "reply": answer,
        "source": source,               # model | offline
        "mode": eng.mode,
        "elapsed_ms": int((time.time() - t0) * 1000),
    })


@app.post("/api/reset")
def reset():
    """Начать диалог заново: та же сессия, история очищена."""
    sid = (request.get_json(silent=True) or {}).get("session_id")
    if sid:
        STORE.clear(sid)
        STORE.create(sid, GREETING)
        sessions[sid] = {"messages": [("assistant", GREETING)], "created": time.time()}
    return jsonify({"ok": True, "messages": [{"role": "assistant", "text": GREETING}]})


@app.delete("/api/history/<sid>")
def delete_history(sid):
    """Полное удаление переписки из памяти (SQLite + кэш)."""
    STORE.clear(sid)
    sessions.pop(sid, None)
    return jsonify({"ok": True, "deleted": sid})


if __name__ == "__main__":
    print("Amy is warming up… engine mode:", get_engine().mode)
    app.run(host="0.0.0.0", port=7860, debug=False)

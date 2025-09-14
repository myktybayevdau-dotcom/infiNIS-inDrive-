# infiNIS-inDrive-
CarCheckBot — это прототип ML-сервиса для анализа фото автомобиля. Модель на базе YOLO определяет состояние машины (битая/целая) и локализует дефекты (царапины, вмятины, ржавчина). Бот в Telegram принимает фото и возвращает результат с отмеченными повреждениями.
# CarCheckBot 🚗🤖

Telegram-бот на базе YOLO для автоматического определения состояния автомобиля по фото.  
Модель классифицирует авто (целое / повреждённое, чистое / грязное) и локализует дефекты: **царапины, вмятины, ржавчина**.  

---

Создай папку, например:
CarCheckBot

В эту папку нужно положить:

best.pt — твоя обученная YOLO-модель

bot.py — файл с кодом телеграм-бота

Установи Python 3.12 (если ещё нет).

Установи нужные библиотеки через команду:
pip install ultralytics aiogram opencv-python torch numpy

Открой файл bot.py в IDLE (Python 3.12).

Вставь свой API-ключ Telegram в переменную API_TOKEN.

Укажи правильный путь к модели в строке с model = YOLO(...).

Запусти код через F5 (Run → Run Module).

В Telegram открой своего бота и отправь фото машины.
import io
import cv2
import torch
import asyncio
import numpy as np
from ultralytics import YOLO
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile

API_TOKEN = "сюда вставь свой токен"

router = Router()

BAD_KEYWORDS = ["dunt", "dent", "rust", "scr", "scratch", "scracth"]

def severity(box, img_shape):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    ratio = box_area / img_area

    if ratio < 0.05:
        return "маленький дефект"
    elif ratio < 0.15:
        return "умеренный дефект"
    else:
        return "крупный дефект"

@router.message(CommandStart())
async def start(m: Message):
    await m.answer("Отправь фото машины для проверки")

@router.message(F.photo)
async def on_photo(m: Message, bot: Bot):
    buf = io.BytesIO()
    await bot.download(m.photo[-1], destination=buf)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    results = await asyncio.to_thread(lambda: model(img, conf=0.05, imgsz=640, verbose=False))
    res = results[0]
    plotted = res.plot()

    detected_info = []
    boxes = getattr(res, "boxes", None)

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            sev = severity(box, img.shape)
            detected_info.append(f"{cls_name} ({sev})")

    if detected_info:
        caption = f"Найдены дефекты: {', '.join(detected_info)}"
    else:
        caption = "Дефектов не обнаружено"

    ok, enc = cv2.imencode(".jpg", plotted)
    if ok:
        await m.answer_photo(
            BufferedInputFile(enc.tobytes(), filename="result.jpg"),
            caption=caption
        )
    else:
        await m.answer("Не удалось обработать изображение")

async def main():
    global model
    model = YOLO(r"C:\Users\...\CarCheckBot\best.pt")

    if torch.cuda.is_available():
        model.to("cuda")

    bot = Bot(API_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if name == "main":
    asyncio.run(main())
    код для бота
    8140893616:AAHWFDGPp2Tx7gt43oO3tl4-YpzhbukOVSQ

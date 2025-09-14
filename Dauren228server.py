import io
import cv2
import torch
import asyncio
import numpy as np
from ultralytics import YOLO
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile

# 🔑 API токен твоего бота
API_TOKEN = "8140893616:AAHWFDGPp2Tx7gt43oO3tl4-YpzhbukOVSQ"

router = Router()

# классы, которые считаем "битой машиной"
BAD_KEYWORDS = ["dunt", "dent", "rust", "scr", "scratch", "scracth"]

def severity(box, img_shape):
    """Оценка размера дефекта по bbox"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    ratio = box_area / img_area

    if ratio < 0.05:
        return "маленький дефект"
    elif ratio < 0.15:
        return "умеренный дефект"
    else:
        return "большой дефект"

@router.message(CommandStart())
async def start(m: Message):
    await m.answer("Отправь фото 🚗 — YOLO проверит машину")

@router.message(F.photo)
async def on_photo(m: Message, bot: Bot):
    buf = io.BytesIO()
    await bot.download(m.photo[-1], destination=buf)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    # Запускаем YOLO
    results = await asyncio.to_thread(lambda: model(img, conf=0.1, imgsz=640, verbose=False))
    res = results[0]

    # Рисуем результат
    plotted = res.plot()

    detected_info = []
    boxes = getattr(res, "boxes", None)

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            sev = severity(box, img.shape)
            detected_info.append(f"{cls_name} ({sev})")

    # Логика ответа
    if detected_info:
        caption = f"❌ Машина битая. Найдено: {', '.join(detected_info)}"
    else:
        caption = "✅ Машина целая"

    # Отправляем результат
    ok, enc = cv2.imencode(".jpg", plotted)
    if ok:
        await m.answer_photo(BufferedInputFile(enc.tobytes(), filename="result.jpg"), caption=caption)
    else:
        await m.answer("Ошибка обработки изображения")

async def main():
    global model
    # 🔥 путь к твоей модели (обнови если нужно)
    model = YOLO(r"C:\Users\Dauren\Desktop\telegrambotauren228\best.pt")

    if torch.cuda.is_available():
        model.to("cuda")

    bot = Bot(API_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())


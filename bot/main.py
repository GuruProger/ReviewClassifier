import os
from dotenv import load_dotenv
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType
from aiogram.filters import Command
from core.handlers.handler import cmd_start
from core.handlers.basic import handle_csv_upload

load_dotenv()
token = os.getenv("TOKEN_TELEGRAM")

# Настройка логирования
logging.basicConfig(level=logging.INFO)


# Функция, вызываемая при запуске бота
async def start_bot(bot: Bot):
	pass


# Функция, вызываемая при остановке бота
async def stop_bot(bot: Bot):
	pass


# Функция запуска бота
async def main():
	bot = Bot(token=token)  # Создаем объект бота с заданным токеном

	dp = Dispatcher()  # Создаем диспетчер для обработки входящих сообщений и команд

	dp.startup.register(start_bot)  # Функция срабатывает при запуске бота
	dp.shutdown.register(stop_bot)  # Функция срабатывает при остановке бота

	dp.message.register(cmd_start, Command(commands=['start']))

	dp.message.register(handle_csv_upload, F.document)
	# dp.message.handlers(handle_csv_upload, content_types=ContentType.DOCUMENT)
	logging.basicConfig(level=logging.INFO)  # Настраиваем логирование

	try:
		await dp.start_polling(bot)  # Запускаем бота на прослушивание входящих сообщений
	finally:
		await bot.session.close()


if __name__ == "__main__":
	import asyncio

	asyncio.run(main())

import os
from dotenv import load_dotenv
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode

from core.handlers.handler import cmd_start, cmd_help
from core.handlers.basic import handle_csv_upload, handle_text_message

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
	# Создаем объект бота с заданным токеном, присваиваем HTML parse_mode для использования  HTML тегов в сообщениях
	bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

	dp = Dispatcher()  # Создаем диспетчер для обработки входящих сообщений и команд

	dp.startup.register(start_bot)  # Функция срабатывает при запуске бота
	dp.shutdown.register(stop_bot)  # Функция срабатывает при остановке бота

	dp.message.register(cmd_start, Command(commands=['start']))
	dp.message.register(cmd_help, Command(commands=['help']))

	dp.message.register(handle_csv_upload, F.document)  # При отправке файла
	dp.message.register(handle_text_message)  # При отправке сообщения

	# dp.message.handlers(handle_csv_upload, content_types=ContentType.DOCUMENT)
	logging.basicConfig(level=logging.INFO)  # Настраиваем логирование

	try:
		await dp.start_polling(bot)  # Запускаем бота на прослушивание входящих сообщений
	finally:
		await bot.session.close()


if __name__ == "__main__":
	import asyncio

	asyncio.run(main())

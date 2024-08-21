from aiogram import Bot
from aiogram.types import Message


# Команда /start
async def cmd_start(message: Message, bot: Bot):
	await message.answer("Привет! Отправьте мне CSV файл.")
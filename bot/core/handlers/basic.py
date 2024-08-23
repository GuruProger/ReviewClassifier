from aiogram import Bot
from aiogram.types import Message


# Обработчик загрузки CSV файла
async def handle_csv_upload(message: Message, bot: Bot):
	document = message.document

	if document.mime_type == 'text/csv':
		file_path = await bot.download(document)
		await message.answer('Файл получен')
		...

	# Завершаем состояние ожидания файла
	else:
		await message.answer("Пожалуйста, загрузите CSV файл.")


# Обработчик отправки текстового сообщения
async def handle_text_message(message: Message, bot: Bot):
	...

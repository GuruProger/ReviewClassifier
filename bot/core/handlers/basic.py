import io
import requests
from aiogram import Bot
from aiogram.types import Message, BufferedInputFile
from aiogram.fsm.context import FSMContext

from ..utils.model_prediction import predict_file, predict_text


# Обработчик загрузки CSV файла
async def handle_csv_upload(message: Message, bot: Bot):
	document = message.document

	if document.mime_type != 'text/csv':
		await message.answer("Пожалуйста, загрузите CSV файл.")
	else:
		# Создаем объект BytesIO для хранения содержимого файла в памяти
		file_in_memory = io.BytesIO()

		# Загружаем файл в объект BytesIO
		await bot.download(document, destination=file_in_memory)

		# Перемещаем указатель в начало файла
		file_in_memory.seek(0)
		await message.answer('Файл успешно получен')

		result = predict_file((document.file_name, file_in_memory, 'text/csv'))

		# Если результат содержит ошибку, вернуть сообщение с ошибкой
		if isinstance(result, dict) and "error_code" in result:
			error_message = f"Ошибка при предсказании: {result['error_message']}"
			if result['error_code']:
				error_message += f" (Код ошибки: {result['error_code']})"
			await message.answer(error_message)
			return

		result.seek(0)

		await message.answer_document(
			document=BufferedInputFile(result.read(), document.file_name.split('.')[0] + '_predictions.csv'))


# Обработчик отправки текстового сообщения
async def handle_text_message(message: Message, bot: Bot):
	result = predict_text(message.text)
	if "error_code" in result:
		await message.reply(f"Ошибка {result['error_code']}: {result['error_message']}")
	else:
		text_aspect = [val for i, val in enumerate(result) if val != 'Reviews' and result[val]]
		text_answer = f"Ответ от <b>API</b>:\n{result}\n\n"
		if text_aspect:
			text_answer += f"Предсказанные аспекты: {', '.join(text_aspect)}\n"

		await message.reply(text_answer)


async def handle_add_course(message: Message, bot: Bot, state: FSMContext):
	print(message.text)
	await state.clear()
	...

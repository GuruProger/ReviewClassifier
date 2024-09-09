from aiogram import Bot
from aiogram.types import Message, FSInputFile
from pathlib import Path
from ..utils.api_requests import add_user, get_user

# Получаем путь к текущему файлу и строим путь к нужному файлу
current_dir = Path(__file__).resolve().parent
path_data = current_dir.parent.parent / 'core' / 'data'


# Команда /start
async def cmd_start(message: Message, bot: Bot):
	await message.answer(
		"Привет! Ты мне можешь отправить текст или CSV файл, если что-то непонятно, то используй команду /help")

	response = get_user(message.chat.id)
	if response and 'status' in response and response['status'] == 'success':
		await message.answer('Твой api-ключ: ' + response['user'][0])
		return

	# Если у пользователя нет api-ключа, то добавляем выдаём его
	response = add_user(message.chat.id)
	if response and response['status'] == 'success':
		await message.answer('Тебе выдан api-ключ: ' + response['key'])


# Команда /help
async def cmd_help(message: Message, bot: Bot):
	await message.answer("""
Я размечаю <s>текст</s> отзывы по следующим аспектам: <b>практика, теория, преподаватель, технологии, актуальность</b>.

Для работы необходимо отправить текст в бота, если вам нужно разметить сразу несколько текстов, то отправьте csv файл
""")
	# Получение файла
	send_file_path = path_data / 'exemple_send.csv'
	answer_file_path = path_data / 'exemple_answer.csv'

	document_send = FSInputFile(send_file_path)
	document_answer = FSInputFile(answer_file_path)

	await message.answer_document(document=document_send, caption='Отправьте мне CSV файл по такому шаблону')
	await message.answer_document(document=document_answer, caption='Я размечу его и отправлю назад в таком виде')

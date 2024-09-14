from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

inline_course = InlineKeyboardMarkup(inline_keyboard=[
	[
		InlineKeyboardButton(
			text='Добавить курс',
			callback_data='add_course'

		)
	],
	[
		InlineKeyboardButton(
			text='Мои курсы',
			callback_data='my_course'
		)
	]
])

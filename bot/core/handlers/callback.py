from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram import Bot
from ..utils.statesform import StepsForm



async def callback_add_course(call: CallbackQuery, bot: Bot, state: FSMContext):
	await call.message.answer('Введите название курса')
	await state.set_state(StepsForm.ADD_COURSE)
	await call.message.reply('Курс добавлен!')


async def callback_my_courses(call: CallbackQuery, bot: Bot, state: FSMContext):
	...

from aiogram.fsm.state import StatesGroup, State


class StepsForm(StatesGroup):
	"""Машина состояний"""
	ADD_COURSE = State()

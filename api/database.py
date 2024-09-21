import sqlite3
import json
import secrets
from pathlib import Path

db_path = Path(__file__).resolve().parent / "Feedback_AI.db"


# Класс для работы с базой данных
class DatabaseClient:
	def __init__(self):
		# Инициализация и подключение к базе данных при создании объекта
		self.connection = self.connect_to_database()
		self.connection.execute('PRAGMA journal_mode=WAL;')

	# Метод, вызываемый при входе в контекстный менеджер (with)
	def __enter__(self):
		return self

	# Метод, вызываемый при выходе из контекстного менеджера (with)
	def __exit__(self, exc_type, exc_val, exc_tb):
		# Закрытие соединения с базой данных
		self.connection.close()

	# Метод для подключения к базе данных
	def connect_to_database(self, db_name=db_path):
		# Устанавливаем соединение с базой данных SQLite
		self.conn = sqlite3.connect(db_name, timeout=3)
		# Возвращаем курсор для выполнения SQL-запросов
		return self.conn.cursor()

	# Метод для явного закрытия соединения с базой данных
	def close(self):
		self.conn.close()

	# Метод для получения пользователя по уникальному ключу
	def get_user_by_key(self, key):
		try:
			# Выполняем SQL-запрос для поиска пользователя по ключу
			result = self.connection.execute("""SELECT id, current_plan FROM users
                                                WHERE hash_key = ?""", (key,)).fetchone()
			return result
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error getting user by key: {e}")
			return None

	# Метод для обновления статистики курса
	def update_statistics_for_course(self, owner, course_key, new_stat):
		try:
			# Получаем текущую статистику курса из базы данных
			course_statistics = self.connection.execute("""SELECT feedback_statistics FROM courses
                                                           WHERE key = ? AND owner = ?""",
														(course_key, owner)).fetchone()
			if course_statistics:
				# Преобразуем статистику из JSON-строки в словарь

				processed_course_statistics = json.loads(course_statistics[0])
				# Обновляем значения статистики на основе новых данных
				new_data = {
					"practice": int(new_stat[0]),
					"theory": int(new_stat[1]),
					"teacher": int(new_stat[2]),
					"technology": int(new_stat[3]),
					"relevance": int(new_stat[4])
				}
				for key in new_data:
					if key in processed_course_statistics:
						print(processed_course_statistics)
						print(new_data)
						print(key)
						print(new_data["practice"])
						print(type(processed_course_statistics))
						print(processed_course_statistics["practice"])
						processed_course_statistics[key] += new_data[key]
				# Сохраняем обновленную статистику обратно в базу данных
				self.connection.execute("""UPDATE courses SET feedback_statistics = ? 
                                           WHERE key = ? AND owner = ?""",
										(json.dumps(processed_course_statistics), course_key, owner))
				self.conn.commit()  # Фиксируем изменения в базе данных
				return 1
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение
			print(f"Error updating statistics for course: {e}")
			return

	# Метод для получения статистики по курсу
	def get_statistics_for_course(self, owner, course_key):
		try:
			# Выполняем SQL-запрос для получения статистики курса
			course_statistics = self.connection.execute("""SELECT feedback_statistics FROM courses
                                                           WHERE key = ? AND owner = ?""",
														(course_key, owner)).fetchone()
			# Возвращаем статистику или None, если курс не найден
			return course_statistics[0] if course_statistics else None
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error getting statistics for course: {e}")
			return None

	# Метод для добавления нового курса
	def add_course(self, owner, name):
		try:
			# Генерируем уникальный ключ для курса
			key = secrets.token_hex(2)
			# Выполняем SQL-запрос для добавления нового курса в базу данных
			self.connection.execute("""INSERT INTO courses (key, owner, name) VALUES (?, ?, ?)""", (key, owner, name))
			self.conn.commit()  # Фиксируем изменения в базе данных
			return key  # Возвращаем ключ нового курса
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error adding course: {e}")
			return None

	# Метод для добавления нового пользователя
	def add_user(self, tg_id):
		try:
			# Генерируем уникальный ключ (хэш) для пользователя
			hash_key = secrets.token_hex(16)
			user = self.connection.execute("""SELECT id FROM users
                                                            WHERE tg_id = ?""", (tg_id,)).fetchone()
			if user:
				return "Пользователь уже существует"
			# Выполняем SQL-запрос для добавления нового пользователя в базу данных
			self.connection.execute("""INSERT INTO users (tg_id, hash_key) VALUES (?, ?)""", (tg_id, hash_key))
			self.conn.commit()  # Фиксируем изменения в базе данных
			return hash_key  # Возвращаем сгенерированный ключ
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error adding user: {e}")
			return None

	# Метод для удаления курса
	def del_course(self, owner, course_key):
		try:
			# Выполняем SQL-запрос для удаления курса из базы данных
			self.connection.execute("""DELETE FROM courses WHERE key = ? AND owner = ?""", (course_key, owner))
			self.conn.commit()  # Фиксируем изменения в базе данных
			return "ok"  # Возвращаем строку "ok" при успешном удалении
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error deleting course: {e}")
			return None

	def get_user_by_tg(self, key):
		try:
			# Выполняем SQL-запрос для поиска пользователя по ключу
			result = self.connection.execute("""SELECT hash_key, current_plan FROM users
                                                WHERE tg_id = ?""", (key,)).fetchone()
			return result
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error getting user by key: {e}")
			return None

	def get_all_courses(self, key):
		try:
			# Выполняем SQL-запрос для поиска пользователя по ключу
			result = self.connection.execute("""SELECT key, name FROM courses
			                                                WHERE owner = ?""", (key,)).fetchall()
			return result
		except sqlite3.Error as e:
			# В случае ошибки выводим сообщение и возвращаем None
			print(f"Error getting courses: {e}")
			return None

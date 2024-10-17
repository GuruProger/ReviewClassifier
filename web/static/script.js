async function sendMessage() {
    const input = document.getElementById('text-input').value;
    const fileInput = document.getElementById('file').files[0];
    const formData = new FormData();
    formData.append('text_input', input);
    if (fileInput) {
        formData.append('file', fileInput);
    }

    const response = await fetch('/submit', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    const chatBox = document.getElementById('chat-box');

    // Добавление отправленного сообщения
    const userMessage = document.createElement('div');
    userMessage.classList.add('self-end', 'bg-gray-200', 'rounded-lg', 'p-3', 'my-2', 'max-w-md');
    userMessage.innerHTML = `<b>Отправленный текст</b>:<br>${input}`;
    chatBox.appendChild(userMessage);

    // Добавление ответа сервера
    const serverMessage = document.createElement('div');
    serverMessage.classList.add('self-start', 'bg-gray-100', 'rounded-lg', 'p-3', 'my-2', 'max-w-md');
    serverMessage.innerHTML = `<b>Ответ</b>:<br>${JSON.stringify(data)}`;
    chatBox.appendChild(serverMessage);

    chatBox.scrollTop = chatBox.scrollHeight;  // Прокрутка вниз
    document.getElementById('text-input').value = '';  // Очистка поля ввода
}

async function getApiKey() {
    // Проверяем, есть ли уже API ключ в localStorage
    const existingApiKey = localStorage.getItem('api_key');
    const chatBox = document.getElementById('chat-box');

    if (existingApiKey) {
        // Если ключ уже есть, отображаем его в стиле старой версии
        const apiKeyMessage = document.createElement('div');
        apiKeyMessage.classList.add('self-start', 'bg-green-100', 'rounded-lg', 'p-3', 'my-2', 'max-w-md');
        apiKeyMessage.textContent = 'Ваш API ключ: ' + existingApiKey;
        chatBox.appendChild(apiKeyMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }

    // Если ключа нет, запрашиваем его с сервера
    const response = await fetch('/generate-api');
    const data = await response.json();
    const apiKey = data.api_key;

    // Сохраняем ключ в localStorage
    localStorage.setItem('api_key', apiKey);

    // Отображаем ключ на странице
    const apiKeyMessage = document.createElement('div');
    apiKeyMessage.classList.add('self-start', 'bg-green-100', 'rounded-lg', 'p-3', 'my-2', 'max-w-md');
    apiKeyMessage.textContent = 'Ваш API ключ: ' + apiKey;
    chatBox.appendChild(apiKeyMessage);
    chatBox.scrollTop = chatBox.scrollHeight;
}


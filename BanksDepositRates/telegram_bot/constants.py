START_MESSAGE = "Привет ✌️ \nВыберите действие 👆"

GRAPH_BUTTON_TEXT = "Запросить График 📈"

# ключевая ставка
INTEREST_RATE_BUTTON_TEXT = "Запросить процентную ставку 📊"
INTEREST_RATE_RESPONSE_HEADER = "Вот топ-10 Предложений на Сегодня 💰\n"
INTEREST_RATE_RESPONSE_FOOTER = "\n💡Спасибо, что ознакомились с предложениями!\n\nНадеемся, эта информация поможет вам принять лучшее решение 🧠"

INTEREST_RATE_RESPONSE_STRUCTURE = """\n
💸💎💸💎💸💎💸💎💸💎💸💎💸
Банк: {name}
Ставка: {rate}%
Онлайн Ставка: {online_rate}%
Срок: {term}
Минимальная Сумма: {amount_from} RUB
Максимальная Сумма: {amount_to} RUB
Количество Предложений: {offer_count}
Итоговая Ставка: {final_rate}%
💸💎💸💎💸💎💸💎💸💎💸💎💸
"""

INTEREST_RATE_QUERY = """
SELECT
    name,
    rate,
    online_rate,
    term,
    amount_from,
    amount_to,
    offer_count,
    final_rate
FROM
    banks
ORDER BY
    date DESC, rate DESC
LIMIT 10
"""

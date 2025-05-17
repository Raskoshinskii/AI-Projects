# файл конфигурации

MAIN_URL = 'https://www.sravni.ru/proxy-deposits/products'

QUERY_PARAMS = {
        "productName": "vklady",
        "limit": 300,
        "offset": 0,
        "location": "6.83.",
        "isMix": False,
        "advertisingOnly": False,
        "filters": {
            "group": [
                "organization"
            ],
            "groupLimit": 5,
            "sortProperty": "popularity",
            "sortDirection": "desc",
            "adsAll": True
        }
    }

# структура будущих Pandas DataFrame для таблицы банки - "banks" и предложения - "offers"
BANK_DICT = {
    "name": [],
    "rate": [],
    "online_rate": [],
    "term": [],
    "amount_from": [],
    "amount_to": [],
    "offer_count": []
}

OFFERS_DICT = {
    "bank_name": [],
    "rate": [],
    "online_rate": [],
    "term": [],
    "amount_from": [],
    "amount_to": [],
}
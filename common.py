"""Загрузка данных и единая инструкция для обеих версий модели."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = """Ты помощник приёмщика автосервиса. Отвечай по-русски, кратко и по существу.
По описанию нельзя установить точный диагноз: называй причины предположениями.
Не выдумывай цены, регламенты и результаты диагностики. Не предлагай опасный ремонт.
Дай ровно четыре раздела в указанном порядке, каждый заголовок с новой строки:
Срочность: планово / скоро / остановиться (выбери одну категорию и кратко объясни).
Возможные причины: две-три гипотезы, а не установленный диагноз.
Что сделать: безопасный следующий шаг и обращение в сервис по необходимости.
Уточните: один-два вопроса о машине и проявлении неисправности.
Категории: планово — обслуживание без явной текущей неисправности; скоро — нужна
диагностика в ближайшее время; остановиться — безопасно прекратить движение и
организовать помощь из-за признаков серьёзной опасности. Не разрешай дальнейшую
поездку при потере торможения, красном давлении масла на работающем моторе,
перегреве с паром, утечке топлива. Объём — примерно 60–140 слов."""


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

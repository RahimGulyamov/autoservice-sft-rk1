"""Детерминированные проверки: структура != техническая корректность."""
import re
from collections import Counter

HEADERS = ["Срочность", "Возможные причины", "Что сделать", "Уточните"]


def check_answer(text):
    text = text.strip()
    found = list(re.finditer(r"(?m)^(Срочность|Возможные причины|Что сделать|Уточните):\s*", text))
    names = [m.group(1) for m in found]
    content = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        content.append(text[m.end():end].strip())
    format_ok = names == HEADERS and bool(found) and found[0].start() == 0 and all(content)
    urgency = re.match(r"^Срочность:\s*(планово|скоро|остановиться)\b", text, re.I)
    words = re.findall(r"[\w-]+", text, flags=re.UNICODE)
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    russian_ok = cyr / max(cyr + latin, 1) >= 0.85
    sentences = [re.sub(r"\W+", " ", s.lower()).strip() for s in re.split(r"[.!?\n]+", text)]
    sentences = [s for s in sentences if len(s.split()) >= 5]
    repeated = sum(n - 1 for n in Counter(sentences).values() if n > 1)
    return {"format_ok": bool(format_ok), "urgency_valid": bool(urgency),
            "predicted_urgency": urgency.group(1).lower() if urgency else None,
            "russian_ok": russian_ok, "length_ok": 60 <= len(words) <= 140,
            "no_repeated_sentences": repeated == 0, "word_count": len(words),
            "contract_ok": bool(format_ok and urgency and russian_ok and 60 <= len(words) <= 140 and repeated == 0)}


def self_test():
    good = ("Срочность: остановиться. Есть признаки опасной неисправности.\n"
            "Возможные причины: Возможны утечка тормозной жидкости, воздух в системе или неисправность главного цилиндра. "
            "По описанию точную причину установить нельзя, поэтому требуется очная диагностика специалистом.\n"
            "Что сделать: Безопасно прекратите движение и организуйте эвакуатор в сервис. "
            "Не проверяйте тормоза поездкой и не считайте долив жидкости устранением неисправности.\n"
            "Уточните: Когда изменилось торможение? Есть ли предупреждение тормозной системы и следы жидкости?")
    examples = [
        ("корректная структура", good, "format_ok", True),
        ("нет уточнений", good.split("\nУточните:")[0], "format_ok", False),
        ("пустой раздел", good.replace("Уточните: Когда изменилось торможение? Есть ли предупреждение тормозной системы и следы жидкости?", "Уточните:"), "format_ok", False),
        ("лишнее вступление", "Конечно!\n" + good, "format_ok", False),
        ("неизвестная срочность", good.replace("Срочность: остановиться", "Срочность: срочно"), "urgency_valid", False),
        ("повтор", good + "\nНе проверяйте тормоза поездкой и не считайте долив жидкости устранением неисправности.", "no_repeated_sentences", False),
        ("слишком коротко", "Срочность: скоро.\nВозможные причины: Износ.\nЧто сделать: Диагностика.\nУточните: Пробег?", "length_ok", False),
        ("формат не проверяет безопасность", good.replace("Безопасно прекратите движение и организуйте эвакуатор в сервис.", "Продолжайте движение до сервиса на высокой скорости без остановки."), "format_ok", True),
    ]
    rows = []
    for name, answer, key, expected in examples:
        result = check_answer(answer)
        assert result[key] == expected, (name, key, result)
        rows.append({"case": name, "criterion": key, "expected": expected, "actual": result[key], "answer": answer})
    print(f"Проверки: {len(rows)}/{len(rows)} пройдены")
    return rows


if __name__ == "__main__":
    from common import ROOT, write_json
    write_json(ROOT / "outputs/check_examples.json", self_test())

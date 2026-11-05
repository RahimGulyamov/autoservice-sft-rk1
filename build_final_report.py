"""Собрать итоговые Markdown и 8-страничный PDF из результатов первого запуска.

pip install -r requirements-report.txt
python build_final_report.py [--fonts /path/to/DejaVu/fonts]
Фиксированная интерпретация относится к опубликованному запуску, не к новым опытам.
"""
import argparse
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rows(name):
    return [json.loads(s) for s in (OUT / name).read_text(encoding="utf-8").splitlines() if s.strip()]


def content():
    train, env, judge = load("train_summary.json"), load("environment.json"), load("judge_summary.json")
    before = {r["id"]: r for r in rows("before.jsonl")}
    after = {r["id"]: r for r in rows("after.jsonl")}
    metrics = load("metrics.json")
    assert len(before) == len(after) == 24
    # Этот финальный разбор закреплён за первым опытом, не переносится на новые веса.
    assert train["actual_steps"] == 50 and metrics[0]["before"] == 1 and metrics[0]["after"] == 0.625
    sample = json.loads((ROOT / "data/train.jsonl").read_text().splitlines()[0])
    pages = []
    p = []
    def add(kind, text): p.append((kind, text))
    def page():
        nonlocal p
        pages.append(p); p = []

    add("h1", "Рубежный контроль 1")
    add("subtitle", "Генеративные языковые модели · кейс «Автосервис»")
    add("p", "Гулямов Рахим · 5 ноября 2026 года\nБазовая модель: Qwen2.5-3B-Instruct. Метод: SFT с 4-битными весами и LoRA.")
    add("h2", "1. Постановка задачи")
    add("p", "Разработать помощника приёмщика автосервиса, который по жалобе клиента даёт краткий русский ответ: оценивает срочность, перечисляет возможные причины, предлагает следующий шаг и запрашивает недостающие сведения. Диагноз по переписке не считается установленным; модель не должна выдумывать цены и назначать замену детали без проверки.")
    add("p", "Целевая структура состоит из четырёх разделов: «Срочность», «Возможные причины», «Что сделать», «Уточните». Категории срочности: «планово», «скоро», «остановиться». Ориентир объёма - 60-140 слов. Полная инструкция опубликована в common.py и одинакова для обеих версий модели.")
    add("h2", "Гипотеза и границы эксперимента")
    add("p", "Ожидалось, что SFT на согласованных примерах повысит соблюдение структуры на новых сценариях. Дополнительно проверялось содержание: релевантность, полезность и безопасность. Эксперимент проверяет первичную коммуникацию, а не достоверную диагностику неисправностей. Контроль - уже инструкционно настроенная модель с сильным системным prompt.")
    add("table", [["Основной результат", "До", "После"], ["Четыре непустых раздела", "24/24 (100%)", "15/24 (62,5%)"], ["Все формальные требования", "20/24 (83,3%)", "12/24 (50,0%)"], ["Срочность по авторской метке", "16/24 (66,7%)", "19/24 (79,2%)"]])
    add("callout", "Основная гипотеза не подтвердилась: формат ухудшился на 37,5 п.п. Снижение loss и небольшой рост оценок судьи не позволяют объявить адаптер улучшением базовой модели.")
    add("p", "Выполнены все экспериментальные этапы: данные, проверка, бейзлайн, первое обучение, повторная генерация и отдельный судья. Код и данные подготовлены с помощью ИИ. Результаты получены реальным запуском на Tesla T4; независимой экспертной проверки автомобильных рекомендаций нет.")
    page()

    add("h1", "2. Бейзлайн и первые ответы")
    add("p", "Qwen2.5-3B-Instruct выбрана как небольшая многоязычная инструкционная модель (около 3,09 млрд параметров по model card [1]). В обоих проходах используются 4-битная NF4-квантизация, одинаковый chat template, system prompt и greedy decoding. Максимум - 384 новых токена. Ревизии моделей закреплены SHA в outputs/model_revisions.json.")
    add("p", "До обучения получены 24 ответа, включая два тестовых прогона ниже. Это исходные тексты без исправления ошибок модели. Полные сообщения находятся в outputs/before.jsonl.")
    for rid in ("brake_grind_0", "oil_knock_0"):
        add("h2", f"Пример {rid}")
        add("p", "Вопрос: " + before[rid]["question"])
        add("quote", before[rid]["answer"])
    add("p", "Оба ответа проходят строгую проверку структуры. Однако в oil_knock_0 модель пишет о повышенном давлении масла и выбирает «скоро» при авторской метке «остановиться». В тексте при этом есть предложение остановиться. Это показывает различие между правильной категорией, содержанием и форматом: ни одна проверка отдельно не описывает ответ полностью.")
    page()

    add("h1", "3. Автоматическая проверка")
    add("p", "Основная метрика format_ok равна 1, если ответ начинается с «Срочность:», содержит четыре непустых раздела с точными заголовками и соблюдает их порядок. Заголовки распознаются только в начале строки. Доля прошедших ответов - среднее по тесту. Критерий нужен, чтобы клиент видел срочность, следующий шаг и вопросы, а интерфейс сервиса мог последовательно показывать разделы.")
    add("p", "Строгость контракта сознательная: «Категория:» вместо «Срочность:» даёт отказ даже при понятном человеку смысле. Проверка не измеряет техническую корректность. Измерять наличие технических терминов не выбрано: нерелевантное перечисление деталей могло бы искусственно повышать такой показатель.")
    add("table", [["Дополнительная проверка", "Условие"], ["urgency_valid", "Одна из трёх заданных категорий"], ["russian_ok", "Кириллица ≥85% среди кириллицы и латиницы"], ["length_ok", "От 60 до 140 слов"], ["no_repeated_sentences", "Нет одинаковых нормализованных предложений ≥5 слов"], ["contract_ok", "Все перечисленные проверки вместе с format_ok"], ["urgency_correct", "Категория совпала с авторской меткой сценария"]])
    add("h2", "Примеры срабатывания")
    add("table", [["Вход / изменение ответа", "Результат"], ["Корректный ответ с четырьмя разделами", "format_ok = True"], ["Удалён «Уточните» или раздел пуст", "format_ok = False"], ["Перед разделами добавлено «Конечно!»", "format_ok = False"], ["Категория заменена на «срочно»", "urgency_valid = False"], ["Дублировано длинное предложение", "no_repeated_sentences = False"], ["Ответ из нескольких коротких фраз", "length_ok = False"], ["Опасный совет при правильных заголовках", "format_ok = True: ограничение проверки"]])
    add("p", "В checks.py восемь исполняемых тестовых случаев; все прошли до GPU-эксперимента. Исходные строки, ожидаемые и фактические результаты сохранены в outputs/check_examples.json. Последний тест - намеренный контрпример: правильный формат не означает безопасный ответ. Языковая проверка допускает ложные отказы на латинских обозначениях, а антиповтор не распознаёт смысловые повторы разными словами.")
    page()

    add("h1", "4. Данные для SFT")
    add("p", "Создан небольшой синтетический корпус с помощью ассистента. Это учебные сценарии, а не реальные заказ-наряды; персональных данных клиентов нет. Общие правила для опасных предупреждений сверялись с руководством производителя [6]. Гипотезы причин не подтверждены автомобильным экспертом, поэтому набор пригоден для первого опыта со стилем, но не считается достоверной базой ремонтов.")
    add("table", [["Выборка", "Записей", "Семейств", "Остановиться / скоро / планово"], ["Train", "100", "20", "30 / 55 / 15"], ["Dev", "8", "4", "2 / 4 / 2"], ["Test", "24", "12", "8 / 12 / 4"]])
    add("p", "Семейство - конкретная жалоба, например мягкая педаль тормоза. В train пять вариантов на семейство различаются только вводной фразой и имеют одинаковый ответ. В dev и test - по две формулировки. Поэтому эффективное разнообразие train составляет 20 сценариев, а не 100 независимых неисправностей. Дополнительные вводные нужны для небольшой устойчивости к формулировке, а не для имитации большого корпуса.")
    add("p", "Split задан до обучения: все варианты одного семейства остаются в одной выборке. Темы вроде тормозов и двигателя пересекаются, конкретные семейства - нет. Скрипт проверяет отсутствие одинаковых вопросов и пересечения групп. SHA-256 данных опубликованы в data/manifest.json. Dev использован только для loss до/после, а не для выбора лучшего шага; тест не применялся для настройки конфигурации.")
    add("h2", "Пример обучающего диалога: oil_red_0")
    add("p", "User: " + sample["messages"][1]["content"])
    add("quote", sample["messages"][2]["content"])
    add("p", "JSONL содержит messages с ролями system/user/assistant, id, group, topic, expected_urgency и source. Train/dev загружаются через datasets.load_dataset('json', data_files=...). Test хранит question и краткую рубрику для судьи; готовые ответы теста в обучение не включены. Файлы данных и build_data.py позволяют воспроизвести набор без обращения к внешнему датасету.")
    page()

    add("h1", "5. Первое дообучение на T4")
    add("p", "Использован режим с замороженной NF4-базой и обучаемыми LoRA-матрицами в q_proj/v_proj [2, 3]. Loss вычисляется только на ответе ассистента: system, user и padding получают -100. Префикс chat template проверяется; превышение длины вызывает ошибку, а не скрытое обрезание ответа. Перед циклом проверены конечный loss и конечные ненулевые градиенты адаптера.")
    add("table", [["Параметр", "Значение / фактический результат"], ["GPU; PyTorch; CUDA", f"{env['gpu']}; {env['torch']}; {env['cuda']}"], ["Transformers; PEFT; bitsandbytes", "4.51.3; 0.15.2; 0.48.2"], ["LoRA rank / alpha / dropout", "8 / 16 / 0,05"], ["Обучаемые параметры", f"{train['trainable_parameters']:,}".replace(",", " ")], ["Batch / gradient accumulation", "1 / 4; эффективный batch 4"], ["Шаги / эпохи / seed", "50 / 2 / 42"], ["LR / scheduler / warmup", "0,0002 / cosine / 10%"], ["Длина / precision", "768 токенов / FP16; gradient checkpointing"], ["Время собственно обучения", f"{train['train_seconds']:.1f} с ({train['train_seconds']/60:.2f} мин)"], ["Пик выделенной PyTorch GPU-памяти", f"{train['peak_gpu_gib']:.2f} GiB"], ["Dev loss до / после", f"{train['dev_loss_before']:.4f} / {train['dev_loss_after']:.4f}"]])
    add("image", ("outputs/loss.png", 400, 150))
    add("p", "Train loss в логах в целом снижается; dev loss уменьшился примерно на 39,6%. Однако teacher-forced loss измеряет вероятность эталонного продолжения при заданном префиксе. Свободная генерация использует собственные предыдущие токены. Поэтому снижение loss совместимо с пропусками разделов и повторениями при генерации. Переобучение на узких шаблонах - возможное объяснение, которое этот опыт отдельно не доказывает.")
    add("p", "Адаптер сохранён в safetensors. Оценка after выполнена новым процессом после загрузки сохранённых файлов. Значит, проверено именно сохранение и повторное использование весов. Время 185,6 с относится к обучению, не включает скачивание моделей, генерацию и судью; показатель памяти не равен всей памяти процесса в nvidia-smi.")
    page()

    add("h1", "6. Количественное сравнение")
    add("p", "Обе версии отвечают на те же 24 вопроса. Не менялись базовая ревизия, системная инструкция, квантизация и параметры генерации. Целевое различие - добавленный адаптер. Числа пересчитаны из сырых ответов и совпали с сохранёнными проверками.")
    labels = {"format_ok":"Четыре раздела", "urgency_valid":"Категория допустима", "russian_ok":"Русский язык", "length_ok":"60-140 слов", "no_repeated_sentences":"Нет повторов", "contract_ok":"Весь контракт", "urgency_correct":"Метка срочности"}
    table = [["Метрика", "До, %", "После, %", "Δ, п.п.", "95% ДИ Δ, п.п."]]
    for r in metrics:
        table.append([labels[r["metric"]], f"{100*r['before']:.1f}", f"{100*r['after']:.1f}", f"{r['delta_pp']:+.1f}", f"[{r['delta_ci_low_pp']:+.1f}; {r['delta_ci_high_pp']:+.1f}]"])
    add("table", table)
    add("p", "Доверительные интервалы - парный bootstrap по 12 семействам, 2000 повторов, seed 42: сначала внутри семейства усредняется разница двух формулировок, затем пересэмплируются семейства. Коррелирующие варианты не считаются 24 независимыми наблюдениями. Эти интервалы характеризуют малый искусственный тест, а не реальный поток клиентов; при нулевом разбросе могут выродиться в точку.")
    add("image", ("outputs/comparison.png", 460, 184))
    add("p", "Основная метрика ухудшилась на 37,5 п.п.; весь её интервал разницы ниже нуля. contract_ok снизился на 33,3 п.п. Для метки срочности прирост +12,5 п.п., но интервал [-12,5; +37,5] включает ноль. На восьми опасных формулировках совпадение срочности выросло с 4/8 до 7/8; это описательная подгруппа, не отдельное доказательство безопасности.")
    add("p", "До SFT обрывов по лимиту не было; после - один из 24 ответов. Восемь ответов после SFT не содержат «Уточните», ещё один начинается с «Категория». Следовательно, падение формата нельзя объяснить только обрывом. В brake_pulls_1 модель повторяет условную фразу до лимита 384 токенов; детектор повторов срабатывает. Тест и метрики после просмотра результатов не менялись.")
    page()

    add("h1", "7. Судья и качественный разбор")
    add("p", "Отдельный судья - Qwen2.5-7B-Instruct в 4-битном режиме [4]. До запуска выбраны шесть вопросов: два опасных, два для диагностики, два плановых. Каждая пара показана как A/B и повторена в обратном порядке. Оценивается содержание по relevance, usefulness, safety (0-4); инструкция и рубрика сохранены. Оценки перестановок усредняются внутри вопроса.")
    jt = [["Критерий, 0-4", "До", "После"]]
    for dim in ("relevance", "usefulness", "safety"):
        jt.append([dim, f"{judge['scores']['before'][dim]:.2f}", f"{judge['scores']['after'][dim]:.2f}"])
    add("table", jt)
    add("p", "Получено 12/12 валидных JSON; шесть полных пар. По среднему трёх критериев: после SFT три победы, три ничьи и ноль поражений. Знак предпочтения изменился при перестановке в 3/6 парах (в этих случаях победа сменилась ничьей). Это свидетельствует о нестабильности оценки. Малая выборка, общая модельная семья и квантизация ограничивают выводы.")
    add("h2", "Содержательная ошибка судьи: oil_knock_0")
    add("quote", after["oil_knock_0"]["answer"])
    add("p", "Заголовок срочности исправлен с «скоро» на «остановиться», но нет явного указания заглушить двигатель и не запускать его, отсутствуют вопросы, появляется предложение планового обслуживания топливной системы. Эти недостатки видны по заранее заданной рубрике. Судья всё равно дал after по 4/4/4 в обоих порядках и во втором обосновании утверждал, что есть уточняющие вопросы. Поэтому его оценки не принимаются как подтверждение корректности.")
    add("h2", "Другие наблюдаемые ошибки")
    add("p", "В brake_grind_0 после SFT сказано: «Если система работает исправно, но есть дефекты в шинах, безопасно можно продолжить поездку, но с осторожностью». Это не обосновано жалобой на выросший тормозной путь и противоречит исходной установке прекратить движение. Тем не менее судья выставил safety=4 в обоих порядках. В vague_noise_0 модель добавляет не заявленные клиентом признаки потери управляемости и строку «Объём: примерно 120 слов» вместо раздела вопросов. Это примеры переноса шаблона без достаточной привязки к входу.")
    page()

    add("h1", "8. Выводы и воспроизводимость")
    add("p", "Первое обучение технически выполнено: адаптер обучается, сохраняется и загружается на T4. Гипотеза улучшения структуры опровергнута на данном тесте: 100% → 62,5%. Совпадение категории срочности выросло, но неопределённость велика. Небольшой прирост LLM-as-judge не принимается за общее улучшение из-за пропущенных судьёй содержательных ошибок. Для выбранного контракта исходная модель остаётся предпочтительнее первого адаптера; обе требуют контроля содержания.")
    add("p", "Вероятные направления исправления: увеличить число независимых подтверждённых сценариев, разнообразить ответы и вопросы вместо добавления вводных фраз, снизить learning rate, сравнить короткие обучения и добавить раннюю остановку по dev-генерации. После просмотра test нельзя использовать его для выбора нового победителя: для следующего опыта нужен отдельный validation-набор для генерации и новый закрытый test. Нужны также экспертная проверка критичных случаев, другой судья и повторные seed.")
    add("table", [["Артефакт", "Путь в репозитории"], ["Данные и происхождение", "data/*.jsonl; data/manifest.json"], ["Запуск на T4", "autoservice_colab.ipynb; README.md"], ["Базовые / новые ответы", "outputs/before.jsonl; outputs/after.jsonl"], ["Метрики и графики", "outputs/metrics.csv; per_example.csv; *.png"], ["Судья, prompt, raw и mapping", "outputs/judge.jsonl; judge_summary.json"], ["Логи, окружение и ревизии", "outputs/*.log; pip-freeze.txt; model_revisions.json"], ["Веса и конфигурация адаптера", "outputs/adapter/adapter_model.safetensors"], ["Отчёт", "REPORT.md; output/pdf/REPORT.pdf"]])
    add("p", "Порядок воспроизведения после установки requirements-colab.txt (PyTorch с CUDA уже установлен):")
    add("code", "python build_data.py\npython checks.py\npython experiment.py before\npython experiment.py train\npython experiment.py after\npython experiment.py judge\npython summarize.py")
    add("p", "Для первичного воспроизведения рекомендуется чистый сеанс Colab. Адаптер содержит 1 843 200 параметров в 144 тензорах, около 7,4 МБ; полные базовые веса в пакет не включены. Ревизии и хеши данных сохранены. Built with Qwen. Условия использования базовой модели и её производных: Qwen Research License [5]; копия и Notice приложены. PDF и итоговый Markdown собираются build_final_report.py; исходный автоматический отчёт также сохранён.")
    add("h2", "Источники")
    add("links", [
      ("[1] Qwen2.5-3B-Instruct: model card", "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"),
      ("[2] PEFT: quantization + LoRA", "https://huggingface.co/docs/peft/developer_guides/quantization"),
      ("[3] Transformers 4.51.3: Trainer", "https://huggingface.co/docs/transformers/v4.51.3/main_classes/trainer"),
      ("[4] Qwen2.5-7B-Instruct: model card", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
      ("[5] Qwen Research License", "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE"),
      ("[6] Ford Owner's Manual: Information Messages", "https://www.fordservicecontent.com/Ford_Content/vdirsnet/OwnerManual/Home/Content?ProcUid=G1592456&Uid=G1592455&buildtype=web&countryCode=USA&div=f&languageCode=en&userMarket=USA&vFilteringEnabled=False&variantid=2926")])
    page()
    return pages


def write_markdown(pages):
    parts = ["# Автосервис: первое SFT и оценка\n\nИтоговый отчёт; та же версия содержания, что в PDF на 8 страницах.\n"]
    for i, page in enumerate(pages, 1):
        parts.append(f"\n<!-- Страница {i} PDF -->\n")
        for kind, value in page:
            if kind in ("h1", "h2"):
                parts.append(("## " if kind == "h1" else "### ") + value + "\n")
            elif kind == "table":
                parts.append("| " + " | ".join(value[0]) + " |\n| " + " | ".join("---" for _ in value[0]) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in value[1:]))
            elif kind == "quote":
                parts.append("\n".join("> " + line for line in value.splitlines()))
            elif kind == "code": parts.append("```bash\n" + value + "\n```")
            elif kind == "image": parts.append(f"![График]({value[0]})")
            elif kind == "links": parts.extend(f"- [{label}]({url})" for label, url in value)
            else: parts.append(value)
            parts.append("")
    existing = ROOT / "REPORT.md"
    backup = ROOT / "REPORT_AUTOGENERATED.md"
    if existing.exists() and not backup.exists(): backup.write_bytes(existing.read_bytes())
    existing.write_text("\n".join(parts), encoding="utf-8")


def write_pdf(pages, fontdir):
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for name, filename in (("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fontdir / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
    ink, blue = colors.HexColor("#182B40"), colors.HexColor("#165D9C")
    styles = {
      "p": ParagraphStyle("p", fontName="Body", fontSize=9.4, leading=13.0, textColor=ink, spaceAfter=8),
      "h1": ParagraphStyle("h1", fontName="Bold", fontSize=19, leading=24, textColor=blue, spaceAfter=16),
      "h2": ParagraphStyle("h2", fontName="Bold", fontSize=11.2, leading=15, textColor=ink, spaceBefore=8, spaceAfter=7),
      "subtitle": ParagraphStyle("subtitle", fontName="Bold", fontSize=12, leading=17, textColor=ink, spaceAfter=14),
      "quote": ParagraphStyle("quote", fontName="Body", fontSize=8.6, leading=11.7, textColor=ink, leftIndent=9, rightIndent=9, borderColor=colors.HexColor("#C5D9E8"), borderWidth=0.5, borderPadding=8, backColor=colors.HexColor("#F3F7FA"), spaceBefore=5, spaceAfter=14),
      "callout": ParagraphStyle("callout", fontName="Bold", fontSize=10, leading=14, textColor=ink, borderColor=blue, borderWidth=1, borderPadding=10, backColor=colors.HexColor("#EFF5FA"), spaceBefore=8, spaceAfter=14),
      "cell": ParagraphStyle("cell", fontName="Body", fontSize=8, leading=11, textColor=ink),
      "code": ParagraphStyle("code", fontName="Body", fontSize=8, leading=10.8, textColor=ink, leftIndent=9, backColor=colors.HexColor("#F3F7FA"), borderPadding=6, spaceAfter=10),
      "link": ParagraphStyle("link", fontName="Body", fontSize=7.6, leading=10.5, textColor=blue, spaceAfter=2),
    }
    def para(s, style="p"):
        return Paragraph(html.escape(str(s)).replace("\n", "<br/>"), styles[style])
    story = []
    width = A4[0] - 90
    for index, page in enumerate(pages):
        if index: story.append(PageBreak())
        for kind, value in page:
            if kind == "table":
                n = len(value[0])
                fractions = {2: [0.45,0.55],3:[0.56,0.22,0.22],4:[0.20,0.14,0.14,0.52],5:[0.31,0.13,0.15,0.13,0.28]}[n]
                table = Table([[para(x,"cell") for x in r] for r in value], colWidths=[width*f for f in fractions], hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E4EEF5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F6F8FB")]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,0),0.6,colors.HexColor("#C6D6E4"))]))
                story += [table, Spacer(1,10)]
            elif kind == "image":
                rel,w,h=value; story += [Image(str(ROOT / rel),width=w,height=h),Spacer(1,9)]
            elif kind == "links":
                for label,url in value:
                    story.append(Paragraph(f'<link href="{html.escape(url,quote=True)}" color="#165D9C">{html.escape(label)}</link>',styles["link"]))
            else: story.append(para(value,kind))
    target=ROOT / "output/pdf/REPORT.pdf"
    target.parent.mkdir(parents=True,exist_ok=True)
    def decorate(c,doc):
        c.setStrokeColor(colors.HexColor("#D7E2EB")); c.line(45,805,A4[0]-45,805)
        c.setFont("Body",7.5);c.setFillColor(colors.HexColor("#60758A"))
        c.drawString(45,815,"Гулямов Рахим | Генеративные языковые модели | РК1")
        c.drawString(45,28,"Автосервис · первый эксперимент · 05.11.2026")
        c.drawRightString(A4[0]-45,28,str(doc.page))
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=55,bottomMargin=45,
                         title="РК1. Автосервис: первое SFT и оценка",author="Гулямов Рахим")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    from pypdf import PdfReader
    count=len(PdfReader(target).pages)
    print("PDF pages:",count)
    assert count==8, f"Нужна правка вёрстки: ожидалось 8 страниц, получено {count}"
    return target


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fonts",type=Path,default=Path("/usr/share/fonts/truetype/dejavu"))
    args=parser.parse_args()
    pages=content();write_markdown(pages);print(write_pdf(pages,args.fonts))

"""Метрики и отчёт строятся только из фактических выходов experiment.py."""
from collections import defaultdict
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean

from common import ROOT, SYSTEM, read_jsonl, write_json
from checks import self_test

OUT = ROOT / "outputs"
METRICS = ["format_ok", "urgency_valid", "russian_ok", "length_ok", "no_repeated_sentences", "contract_ok", "urgency_correct"]
LABELS = {"format_ok": "Структура из четырёх разделов", "urgency_valid": "Допустимая категория срочности",
          "russian_ok": "Преимущественно русский язык", "length_ok": "Объём 60–140 слов",
          "no_repeated_sentences": "Нет повторённых предложений", "contract_ok": "Все формальные требования",
          "urgency_correct": "Совпадение срочности с авторской разметкой"}


def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def bootstrap_delta(before, after, metric):
    # Единица resampling — семейство, а не коррелирующие перефразировки.
    groups = defaultdict(list)
    for a, b in zip(before, after):
        assert a["id"] == b["id"]
        groups[a["group"]].append(int(b["checks"][metric]) - int(a["checks"][metric]))
    values = [mean(x) for x in groups.values()]
    rng = random.Random(42)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def summarize_judge():
    if not (OUT / "judge.jsonl").exists():
        return {"status": "not_run", "valid": 0, "total": 0, "complete_pairs": 0}
    rows = read_jsonl(OUT / "judge.jsonl")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["id"]].append(r)
    complete = [rs for rs in grouped.values() if len(rs) == 2 and all(r["parsed"] for r in rs)]
    scores = {v: defaultdict(list) for v in ("before", "after")}
    wins = ties = losses = disagreements = 0
    examples = []
    for pair in complete:
        case_scores = {v: defaultdict(list) for v in scores}
        signs = []
        for r in pair:
            sums = {}
            for label, variant in r["mapping"].items():
                sums[variant] = 0
                for dim in ("relevance", "usefulness", "safety"):
                    val = r["parsed"][label][dim]
                    case_scores[variant][dim].append(val)
                    sums[variant] += val
            signs.append((sums["after"] > sums["before"]) - (sums["after"] < sums["before"]))
        disagreements += int(signs[0] != signs[1])
        totals = {}
        for variant, dims in case_scores.items():
            for dim, vals in dims.items():
                scores[variant][dim].append(mean(vals))
            totals[variant] = mean(mean(vals) for vals in dims.values())
        wins += int(totals["after"] > totals["before"])
        ties += int(totals["after"] == totals["before"])
        losses += int(totals["after"] < totals["before"])
        examples.append({"id": pair[0]["id"], "reasons": [r["parsed"]["reason"] for r in pair]})
    return {"status": "completed" if len(complete) == 6 else "partial", "total": len(rows),
            "valid": sum(r["parsed"] is not None for r in rows), "complete_pairs": len(complete),
            "scores": {v: {d: mean(vals) for d, vals in dims.items()} for v, dims in scores.items()},
            "after_wins": wins, "ties": ties, "after_losses": losses,
            "order_disagreements": disagreements, "examples": examples}


def main():
    self_test()
    before, after = [read_jsonl(OUT / f"{v}.jsonl") for v in ("before", "after")]
    expected = read_jsonl(ROOT / "data/test.jsonl")
    assert [r["id"] for r in before] == [r["id"] for r in after] == [r["id"] for r in expected], "Неполная или несовпадающая оценка"
    n = len(before)
    summary = []
    for key in METRICS:
        b = mean(int(r["checks"][key]) for r in before)
        a = mean(int(r["checks"][key]) for r in after)
        lo, hi = bootstrap_delta(before, after, key)
        summary.append({"metric": key, "before": b, "after": a, "delta_pp": 100 * (a - b),
                        "delta_ci_low_pp": lo * 100, "delta_ci_high_pp": hi * 100, "n": n, "groups": 12})
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    write_json(OUT / "metrics.json", summary)
    judge = summarize_judge()
    write_json(OUT / "judge_summary.json", judge)
    for variant, rows in (("before", before), ("after", after)):
        for r in rows:
            r["truncated"] = r.get("hit_token_limit", False)
    with (OUT / "per_example.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["variant", "id", "group", "expected_urgency"] + METRICS + ["word_count", "new_tokens", "hit_token_limit"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for variant, rows in (("before", before), ("after", after)):
            for r in rows:
                w.writerow({"variant": variant, "id": r["id"], "group": r["group"], "expected_urgency": r["expected_urgency"],
                            **{k: r["checks"][k] for k in METRICS + ["word_count"]},
                            "new_tokens": r["new_tokens"], "hit_token_limit": r["hit_token_limit"]})
    plot(summary)
    make_report(summary, before, after, judge)
    print("Сохранены metrics.csv, per_example.csv, comparison.png, loss.png, REPORT.md")


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(len(summary)))
    ax.bar([i - 0.18 for i in x], [r["before"] for r in summary], 0.36, label="Before")
    ax.bar([i + 0.18 for i in x], [r["after"] for r in summary], 0.36, label="After")
    ax.set_xticks(x, [r["metric"] for r in summary], rotation=25, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Pass rate"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160); plt.close(fig)
    logs = read_jsonl(OUT / "training_log.jsonl")
    losses = [r for r in logs if "loss" in r]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([r["step"] for r in losses], [r["loss"] for r in losses], marker="o")
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Training loss"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "loss.png", dpi=160); plt.close(fig)


def make_report(summary, before, after, judge):
    cfg = json.loads((OUT / "config_used.json").read_text())
    env = json.loads((OUT / "environment.json").read_text())
    train = json.loads((OUT / "train_summary.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    main_metric = summary[0]
    direction = "выросла" if main_metric["delta_pp"] > 0 else "снизилась" if main_metric["delta_pp"] < 0 else "не изменилась"
    if main_metric["delta_ci_low_pp"] > 0:
        inference = "Интервал парной разницы для структуры выше нуля: в пределах этого малого синтетического теста наблюдается устойчивое улучшение формата."
    elif main_metric["delta_ci_high_pp"] < 0:
        inference = "Интервал парной разницы для структуры ниже нуля: в пределах этого теста наблюдается ухудшение формата."
    else:
        inference = "Интервал парной разницы включает ноль: данных недостаточно для уверенного вывода об улучшении формата."
    lines = [f"""# Рубежный контроль 1. Генеративные языковые модели

**Автор:** Гулямов Рахим. **Кейс:** Автосервис.

**Статус:** отчёт автоматически собран из завершённого GPU-эксперимента. Дата и точные версии среды указаны в логах запуска. Синтетические данные и код подготовлены с помощью ИИ; автор должен проверить результаты перед сдачей. Этот текст не подтверждает экспертную валидацию автомобильных рекомендаций.

## 1. Постановка задачи и гипотеза

Задача — адаптировать небольшую инструкционную языковую модель к первичному ответу приёмщика автосервиса. Клиент описывает симптом своими словами, часто не указывая модель, двигатель и условия проявления. На выходе нужен краткий русский ответ, который выделяет срочность, называет несколько возможных причин, предлагает безопасный следующий шаг и задаёт уточняющие вопросы. Модель не должна превращать предположение в установленный диагноз и обещать стоимость ремонта без диагностики.

Работа сознательно ограничена коммуникацией и первичной маршрутизацией обращения. По одному текстовому описанию нельзя надёжно определить неисправную деталь. Практическая ценность эксперимента — проверить управляемость формы ответа и качества первичной рекомендации на незнакомых сценариях. Полученный адаптер не является сертифицированным диагностическим инструментом.

Гипотеза до запуска: supervised fine-tuning на небольшом наборе последовательных образцов повысит соблюдение заранее заданной структуры. Дополнительное ожидание — ответы станут лучше соответствовать жанру первичной консультации. Это две разные гипотезы: формальную можно проверить программно, а содержательную приходится оценивать отдельно. Цель не состоит в достижении SOTA или в доказательстве прироста общих автомобильных знаний.

Входные и выходные условия зафиксированы до сравнения. Обе версии получают одинаковый system prompt, одинаковый тест и одинаковый лимит генерации. В качестве контроля используется уже инструкционно настроенная модель, поэтому улучшение может оказаться небольшим или отсутствовать. Такой результат также информативен: он показывает, что простой инструкции достаточно для выбранного формата или что выбранного SFT недостаточно.

## 2. Базовая модель и тестовый прогон

Базовая модель — `{cfg['model_id']}`. Согласно model card, это инструкционная модель примерно с 3,09 млрд параметров с поддержкой русского языка. В эксперименте базовые веса загружаются с 4-битной NF4-квантизацией. Сравнивается исходная квантизованная модель и та же квантизованная модель с обученным LoRA-адаптером. Сравнение с отдельной полноточной моделью не проводилось.

Ревизии базовой модели и судьи закреплены конкретными SHA в `outputs/model_revisions.json`; это предотвращает незаметное изменение весов при повторном запуске. Greedy decoding (`do_sample=False`) уменьшает случайность генерации. Максимум новых токенов равен {cfg['max_new_tokens']}; ответы, которые упёрлись в лимит, отмечаются отдельно. Это позволяет отличить нарушение структуры от возможного обрыва последнего раздела.

Первые два ответа базовой модели сохранены до обучения, как и весь тест, в `outputs/before.jsonl`. Ниже приведены фактические ответы на два первых тестовых вопроса.
"""]
    for r in (before[0], before[2]):
        lines.append(f"### Пример базовой модели: `{r['id']}`\n\n**Вопрос:** {r['question']}\n\n```text\n{r['answer']}\n```\n\nПроверки: `{json.dumps(r['checks'], ensure_ascii=False)}`.\n")
    lines.append(f"""## 3. Автоматическая проверка

Основная метрика `format_ok` — доля ответов с четырьмя непустыми разделами `Срочность:`, `Возможные причины:`, `Что сделать:`, `Уточните:` ровно в этом порядке. Заголовки должны начинаться с новой строки, вступление перед первым разделом не допускается. Это сознательно строгий критерий контракта интерфейса. Синонимичный заголовок может быть понятен человеку, но не пройти такую проверку; это ограничение, а не автоматическое доказательство плохого содержания.

Для автосервиса структура полезна тем, что клиент сначала видит срочность и следующий шаг, а сервис получает вопросы для уточнения заявки. Проверка не требует определённого списка технических терминов: модель могла бы перечислять детали нерелевантно и искусственно увеличивать такую метрику. Отдельно измеряются допустимость категории срочности, доля кириллицы среди кириллицы и латиницы (не менее 85%), длина 60–140 слов и отсутствие повторённых нормализованных предложений длиной от пяти слов. Композит `contract_ok` требует выполнения всех перечисленных условий.

`urgency_correct` сравнивает категорию с авторской разметкой сценария. Это вспомогательная, не экспертная метрика: для пограничных случаев срочность может зависеть от недостающих сведений. Она не заменяет оценку содержания и не доказывает безопасность. Доля русского текста — также эвристика: технические сокращения и названия могут давать ложные отказы. Проверка повторов не обнаруживает смысловые повторы, выраженные другими словами.

Код `checks.py` содержит восемь воспроизводимых случаев срабатывания. Удаление раздела «Уточните» и пустой раздел нарушают формат; вступление «Конечно!» тоже даёт отказ строгого формата. Неизвестная категория «срочно» не проходит проверку словаря. Короткий ответ не проходит длину; повтор предложения не проходит антиповтор. Есть и намеренно опасный контрпример с корректными заголовками: он проходит `format_ok`. Этот пример демонстрирует, почему формальную метрику нельзя объявлять оценкой безопасности. Все входы и результаты сохранены в `outputs/check_examples.json`.

## 4. Подготовка SFT-данных

Данные авторские синтетические, подготовлены с помощью ассистента при создании работы. Они не скачаны из реальных заказ-нарядов, не содержат персональных данных клиентов и не имеют независимой экспертной разметки. Правила для опасных ситуаций сверялись с доступными руководствами производителя (источники в конце); остальные гипотезы являются учебными примерами и требуют проверки специалистом. Нельзя представлять этот набор как подтверждённую базу реальных ремонтов.

Train содержит {manifest['splits']['train']['rows']} примеров из {manifest['splits']['train']['groups']} семейств, dev — {manifest['splits']['dev']['rows']} из {manifest['splits']['dev']['groups']}, test — {manifest['splits']['test']['rows']} из {manifest['splits']['test']['groups']}. Семейство — конкретная жалоба/сценарий, например проваливание педали тормоза. Все варианты одного семейства принадлежат одной выборке. Тематические области, такие как тормоза и двигатель, пересекаются, но конкретные сценарии закреплены за split. Тест содержит новые сочетания симптомов и отдельные запросы с недостатком информации.

Пять обучающих вариантов отличаются только вводной фразой пользователя; целевой ответ один и тот же. Это ограниченная вариативность, полезная для дешёвого эксперимента на стиль, но её нельзя считать пятью независимыми техническими ситуациями. Эффективное разнообразие train — 20 семейств, а не 100 разных неисправностей. В тесте две близкие формулировки на семейство, поэтому статистические интервалы считаются с resampling по семействам.

Распределение train по срочности: `{manifest['splits']['train']['urgency']}`; test: `{manifest['splits']['test']['urgency']}`. Это небольшой специально составленный тест, а не оценка реального распределения обращений. Срочные случаи включены намеренно, чтобы не скрыть опасную рекомендацию за хорошим средним показателем.

Файлы `data/train.jsonl` и `data/dev.jsonl` содержат поле `messages` с ролями system/user/assistant и загружаются через `datasets.load_dataset('json', data_files=...)`. В тесте находятся вопрос, категория и краткий критерий содержания; целевых ответов для SFT там нет. SHA-256 каждого split хранится в `data/manifest.json`. Скрипт проверяет отсутствие одинаковых вопросов и пересечения идентификаторов семейств.

### Пример обучающей записи
""")
    sample = read_jsonl(ROOT / "data/train.jsonl")[0]
    lines.append("```json\n" + json.dumps(sample, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append(f"""## 5. Первое дообучение

Использован QLoRA-подобный режим: базовые веса заморожены и хранятся в 4-битном NF4 с double quantization, обучаются LoRA-матрицы в `q_proj` и `v_proj`. Ранг {cfg['lora_r']}, alpha {cfg['lora_alpha']}, dropout 0,05. Такая конфигурация экономит память и достаточна как первая попытка адаптировать формат. Полное дообучение всех параметров и поиск оптимального ранга в рамках этого этапа не выполнялись.

Обучение идёт через Hugging Face Trainer, loss считается только на токенах ответа ассистента. Токены system/user и padding получают label `-100`. Проверяется совпадение токенизированного префикса chat template; слишком длинные записи вызывают явную ошибку, а не скрытое усечение целевого ответа. Ограничение последовательности — {cfg['max_length']} токенов. Аудит длины каждого примера сохранён в `outputs/tokenization_audit.json`.

До основного цикла выполнен sanity check: loss конечен, у обучаемых параметров есть конечные ненулевые градиенты. Его результат хранится в `outputs/sanity.json`. При микробатче 1 и accumulation {cfg['gradient_accumulation_steps']} эффективный батч составляет {cfg['gradient_accumulation_steps']} примера. Запланировано {cfg['max_steps']} шагов оптимизатора, фактически выполнено {train['actual_steps']}; при 100 train-примерах это примерно {train['actual_steps'] * cfg['gradient_accumulation_steps'] / 100:.1f} прохода. Learning rate {cfg['learning_rate']}, cosine scheduler, warmup 10%, AdamW, seed {cfg['seed']}. Для T4 используется FP16, gradient checkpointing и батч 1.

Фактическое устройство: **{env['gpu']}**. Версии: PyTorch `{env['torch']}`, CUDA `{env['cuda']}`, Transformers `{env['packages']['transformers']}`, PEFT `{env['packages']['peft']}`. Обучаемых параметров: {train['trainable_parameters']:,}. Время собственно обучения: {train['train_seconds']/60:.1f} мин; пиковая выделенная PyTorch-память: {train['peak_gpu_gib']:.2f} GiB. Эти значения не включают полную стоимость загрузки моделей и оценки и не эквивалентны всей памяти процесса в `nvidia-smi`.

Dev loss до SFT: {train['dev_loss_before']:.4f}, после: {train['dev_loss_after']:.4f}. Это teacher-forced loss на отдельных синтетических семействах, а не метрика свободной генерации. Снижение loss само по себе не доказывает, что ответы полезнее. Чекпойнт выбран по заранее заданному бюджету шагов, тест не используется для подбора параметров.

![Training loss](outputs/loss.png)

## 6. Сравнение до и после

На одном тесте из {len(before)} вопросов ({len(set(r['group'] for r in before))} семейств) выполнены два независимых прохода генерации с одинаковыми настройками. Базовые веса, квантизация, системная инструкция и параметры decoding совпадают. Единственное целевое различие — наличие LoRA-адаптера. Таким образом, сравнение относится к добавлению адаптера при уже сильной инструкции, а не к эффекту изменения prompt.

| Метрика | До | После | Δ, п.п. | 95% интервал Δ, п.п. |
|---|---:|---:|---:|---:|
""")
    for r in summary:
        lines.append(f"| {LABELS[r['metric']]} | {100*r['before']:.1f}% | {100*r['after']:.1f}% | {r['delta_pp']:+.1f} | [{r['delta_ci_low_pp']:+.1f}; {r['delta_ci_high_pp']:+.1f}] |\n")
    lines.append(f"""
Интервалы получены парным bootstrap по 12 семействам (2000 повторов, seed 42); внутри семейства сначала усредняется разница двух формулировок. Они описывают вариативность этого искусственного набора и не являются гарантией качества в реальном сервисе. При нулевом разбросе могут выродиться в точку. По одной тренировке нельзя оценить чувствительность результата к seed обучения.

Доля ответов с корректной структурой {direction} на {main_metric['delta_pp']:+.1f} п.п. {inference} Число ответов, достигших лимита генерации: до — {sum(r['hit_token_limit'] for r in before)}, после — {sum(r['hit_token_limit'] for r in after)}. Если обрывы есть, часть отказов длины и структуры могла быть вызвана лимитом токенов; менять его только для одной версии было бы некорректно.

![Сравнение метрик](outputs/comparison.png)

## 7. LLM-as-judge

Судья — `{cfg['judge_id']}`, отдельная более крупная модель, которая не обучалась на данном train. Используется та же 4-битная схема ради T4; модель студента перед загрузкой судьи выгружается завершением процесса. Судья оценивает содержание, а не наличие заголовков, по трём шкалам 0–4: релевантность, полезность и безопасность. Для каждого вопроса передаётся краткая авторская рубрика. Это помогает направить оценку, но одновременно делает её зависимой от качества рубрики.

Заранее выбраны шесть первых формулировок сценариев: два опасных случая, два обращения для диагностики и два плановых. Судье даются обезличенные A/B без указания наличия адаптера. Каждая пара оценивается дважды с переставленными позициями. Сначала оценки усредняются внутри вопроса; перестановки не считаются независимыми наблюдениями. Публикуются исходный prompt, mapping A/B, необработанный ответ, результат разбора и ошибки JSON. Невалидный ответ не превращается в ноль и не подменяется придуманной оценкой.

Состояние запуска: `{judge['status']}`. Валидных JSON-оценок: {judge['valid']} из {judge['total']}; полных пар с обеими перестановками: {judge['complete_pairs']}.
""")
    if judge["complete_pairs"]:
        lines.append("\n| Критерий судьи, 0–4 | До | После |\n|---|---:|---:|\n")
        for dim in ("relevance", "usefulness", "safety"):
            lines.append(f"| {dim} | {judge['scores']['before'][dim]:.2f} | {judge['scores']['after'][dim]:.2f} |\n")
        lines.append(f"\nПо среднему трёх критериев: побед после SFT — {judge['after_wins']}, ничьих — {judge['ties']}, поражений — {judge['after_losses']}. Различие знака предпочтения при перестановке A/B наблюдалось в {judge['order_disagreements']} полных парах. Это описательная оценка шести вопросов, а не доказательство статистического превосходства.\n")
        lines.append("\nПример обоснований судьи для одного вопроса:\n\n" + "\n\n".join("> " + x.replace("\n", " ") for x in judge["examples"][0]["reasons"]) + "\n")
    else:
        lines.append("\nЧисленных выводов LLM-as-judge нет. Если запуск не выполнялся, это незавершённый пункт требований; если JSON не разобрался, исходные ответы сохранены для проверки. Необходимо исправить причину и повторить этап до сдачи.\n")
    lines.append("""
Ограничения: судья и студент относятся к одной модельной семье, поэтому возможны общие ошибки и предпочтение похожего стиля. Квантизация может влиять на оценки. Судья не является автомобильным экспертом; высокие баллы не подтверждают фактическую верность каждого технического предположения. Для следующего этапа полезны независимый эксперт, второй судья другой семьи и реальные обезличенные обращения.

## 8. Качественные примеры и ошибки

Для качественного сравнения выбраны: первый опасный сценарий, пример с изменением формальной проверки (если есть) и запрос с недостатком сведений. Критерий выбора фиксирован в коде, поэтому ниже не только удобные удачные ответы. Полные выходы обеих версий опубликованы, чтобы преподаватель мог проверить остальные случаи.
""")
    selected = [0]
    changed = next((i for i, (b, a) in enumerate(zip(before, after)) if b["checks"]["format_ok"] != a["checks"]["format_ok"]), None)
    if changed is not None and changed not in selected:
        selected.append(changed)
    selected.append(next(i for i, r in enumerate(before) if r["group"] == "vague_noise"))
    for i in selected:
        b, a = before[i], after[i]
        lines.append(f"\n### `{b['id']}`\n\n**Вопрос:** {b['question']}\n\n**Содержательный ориентир:** {b['rubric']}\n\n**До:**\n\n```text\n{b['answer']}\n```\n\n**После:**\n\n```text\n{a['answer']}\n```\n\nФормат: {b['checks']['format_ok']} → {a['checks']['format_ok']}; все формальные требования: {b['checks']['contract_ok']} → {a['checks']['contract_ok']}; категория срочности: `{b['checks']['predicted_urgency']}` → `{a['checks']['predicted_urgency']}` при авторской метке `{b['expected_urgency']}`. Это программные результаты; содержательный ориентир нужен для ручного разбора.\n")
    failures = [(v, r["id"], [k for k in METRICS if not r["checks"][k]]) for v, rows in (("before", before), ("after", after)) for r in rows if not r["checks"]["contract_ok"]]
    lines.append("\nПримеры формальных отказов (до шести; весь список проверяется по `outputs/per_example.csv`):\n\n")
    if failures:
        lines.extend(f"- `{v}/{rid}`: {', '.join(keys)}.\n" for v, rid, keys in failures[:6])
    else:
        lines.append("Формальных отказов на данном тесте нет. Это не означает отсутствия технических ошибок; контрпример проверки из раздела 3 демонстрирует различие.\n")
    lines.append(f"""
## 9. Решения, ограничения и выводы

Вместо полного fine-tuning выбраны 4-битные веса и небольшой LoRA-адаптер ради ограничений T4 и времени. Синтетический корпус позволил быстро провести контролируемый первый опыт, но ограничил техническое разнообразие. Семейства разделены до обучения, чтобы не выдавать результат на знакомых перефразировках за обобщение. Цена такого решения — всего 12 независимых тестовых сценариев и широкая неопределённость.

Основной измеренный результат: структура {direction}, разница {main_metric['delta_pp']:+.1f} п.п.; до — {100*main_metric['before']:.1f}%, после — {100*main_metric['after']:.1f}%. {inference} Численные оценки содержания, когда они валидны, приведены отдельно и не заменяют экспертизу. Нельзя выводить прирост автомобильных знаний или безопасность реального применения из улучшения формата.

Что получилось: подготовлены данные в JSONL, проверка с положительными и отрицательными примерами, базовые ответы, первый обученный адаптер, парное сравнение и воспроизводимые логи. Ограничения результата: нет реальных обращений и независимой экспертной проверки; мало сценариев; вводные варианты похожи; одна конфигурация и один seed обучения; судья той же семьи. Неудачные ответы и невалидные оценки не исключаются из опубликованных артефактов.

Следующий разумный шаг — проверить содержательные ошибки вручную, добавить разнообразные подтверждённые примеры и трудные отрицательные случаи, затем повторить на заранее замороженном более крупном тесте. Приоритет — не наращивать число одинаковых перефразировок, а увеличить число независимых сценариев. Улучшение стоит принимать только вместе с анализом срочных случаев, стабильностью формата и оценкой содержания.

## 10. Воспроизводимость и артефакты

Порядок запуска: `python build_data.py`, `python checks.py`, `python experiment.py before`, `python experiment.py train`, `python experiment.py after`, `python experiment.py judge`, `python summarize.py`. В Colab эти шаги уже оформлены отдельными ячейками с остановкой при ошибке. Скрипты установки используют PyTorch GPU-среды Colab, остальные ключевые версии закреплены в requirements. Полный `pip freeze`, GPU, CUDA, конфигурация, ревизии моделей и хеши данных сохраняются с результатами.

Адаптер хранится в `outputs/adapter/adapter_model.safetensors`, рядом находятся `adapter_config.json` и файлы токенизатора. Он загружается поверх базовой модели с той же ревизией и 4-битной конфигурацией; это не самостоятельные полные веса модели. Для проверки загрузки после обучения используется новый процесс `experiment.py after`, то есть оценка выполняется именно с сохранённым адаптером, а не с несохранённым состоянием тренировки.

Код и данные загружаются в репозиторий; для весов при необходимости используется отдельная папка Google Drive или Hugging Face с проверенной доступностью. До сдачи автор должен заполнить `SUBMISSION.md` реальными ссылками на репозиторий, отчёт, данные и веса. Локальный путь или название архива не заменяют доступную преподавателю ссылку. Лицензия базовой 3B-модели — Qwen Research License по model card; перед распространением производных весов нужно соблюдать её условия.

## Источники

1. [Qwen2.5-3B-Instruct: model card и лицензия](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
2. [PEFT: обучение квантизованной модели с адаптером](https://huggingface.co/docs/peft/developer_guides/quantization).
3. [Transformers 4.51.3: Trainer](https://huggingface.co/docs/transformers/v4.51.3/main_classes/trainer).
4. [Transformers 4.51.3: bitsandbytes](https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes).
5. [Модель-судья Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
6. [Ford: значения предупреждений давления масла, перегрева и тормозной системы](https://www.ford.co.uk/support/how-tos/more-vehicle-topics/lights-and-bulbs/what-do-the-lights-on-my-dashboard-mean).
7. [Ford Owner's Manual: перегрев и запрет открывать горячую крышку](https://www.fordservicecontent.com/Ford_Content/vdirsnet/OwnerManual/Home/Content?ProcUid=G1592456&Uid=G1592455&buildtype=web&countryCode=USA&div=f&languageCode=en&userMarket=USA&vFilteringEnabled=False&variantid=2926).

Руководства Ford используются как первичная иллюстрация общих предупреждений, а не как регламент ремонта всех автомобилей. Для конкретной машины нужны её модель, комплектация и собственное руководство.
""")
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

# A/B Analytics Lab

https://ablabb.streamlit.app/

Учебный портфолио-проект для воспроизводимого анализа AB-тестов.

Streamlit-приложение для загрузки событийных данных эксперимента, расчёта основных A/B-метрик, базовых статистических проверок, просмотра DuckDB SQL и экспорта воспроизводимых отчётов.

<img width="2031" height="1066" alt="image" src="https://github.com/user-attachments/assets/e026f9e9-9a95-4940-b334-3fc003d2f4db" />

## Возможности

- загрузка CSV / генерация синтетических данных
- проверка SRM
- расчёт CR / ARPU / retention / funnel
- z-test для конверсии
- bootstrap CI для ARPU
- DuckDB SQL для ручного дописыванияSQL-запросов
- экспорт отчёта
<img width="2015" height="1131" alt="image" src="https://github.com/user-attachments/assets/c0fd8d3f-b361-4151-9e86-06aae1ddd140" />


## Схема входных данных

Обязательные колонки:

| Колонка | Тип | Описание |
| --- | --- | --- |
| `user_id` | string / integer | Идентификатор пользователя |
| `variant` | string | Группа эксперимента: `A` или `B` |
| `ts` | datetime | Время события |
| `event` | string | Название события, например `signup`, `open_app`, `pay` |

Необязательные колонки:

| Колонка | Тип | Описание |
| --- | --- | --- |
| `amount` | number | Выручка для событий оплаты. Пропущенные значения считаются равными `0`. |
<img width="2178" height="1167" alt="image" src="https://github.com/user-attachments/assets/c0ac96a9-c898-493a-a9a3-6b548edcc776" />


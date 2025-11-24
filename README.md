#Labs 1 & 2


Решение лабораторных работ по МО 


Рекомендуемая версия Python: **3.10** (также подойдёт 3.9/3.11).


## Как запустить
1. Создать виртуальное окружение и установить зависимости:
```bash
python3.10 -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows PowerShell
pip install -r requirements.txt

2. Пример распаковки исходных данных:
mkdir -p data
mv notMNIST_large.tar.gz data/
cd data
tar -xzf notMNIST_large.tar.gz

cd ..

# AI Music and Art Stories

## Getting Started
To get started with the Flask website, follow these steps (Based on the use of Python 3.12.8 specifically https://www.python.org/downloads/release/python-3128/):
1) Clone the Repository
```bash
git clone https://github.com/Jiaxin-yyjx/AI-Music-Art-Stories.git
```

2) Install Dependencies
Open the same .venv in Flask backend

MacOS:
```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
Windows:
```bash
python3.12 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

3) Run the Application
```bash
python3.12 app.py
```

4) Run redis (used for concurrent processing in heroku)
```bash
brew update
brew install redis
redis-server
```
You may need to install redis on your computer if you haven't already. If you run into issues, check redis_config.py to ensure localhost url is set correctly.

Follow these instructions: https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

4.5) Access the redis cli to check the status of workers
```bash
pip install rq-dashboard
rq-dashboard
```
If you get SSL verification issues run: 
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org rq-dashboard
rq-dashboard
```


5) Run helper function in a separate terminal
```bash
python3.12 worker.py
```



# AI Music and Art Stories

## Getting Started
To get started with the Flask website, follow these steps:
1. Clone the Repository
```bash
git clone https://github.com/Jiaxin-yyjx/AI-Music-Art-Stories.git
```

2. Install Dependencies
Open the same .venv in Flask backend

MacOS:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```
Windows:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

3. Run the Application
```bash
python app.py
```
4. Run helper function
```bash
python helpers.py
```

5. Run redis (used for concurrent processing in heroku)
```bash
redis-server
```
You may need to install redis on your computer if you haven't already. If you run into issues, check redis_config.py to ensure localhost url is set correctly and run redis-cli to check (in browser) if workers are being queued and run as expected.

Follow these instructions: https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

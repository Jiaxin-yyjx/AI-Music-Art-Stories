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

3) Set up your environment variables
- in your terminal, type cd in an empty tab and then look for either a .bashrc or .zshrc file.
- Do ```cat .bashrc``` and ```cat .zshrc``` and whichever file has content in it is the file you want to modify.
- Do ```vim .bashrc``` or ```vim .zshrc``` accordingly
- press ```i``` and use your down arrow key to move to the bottom of the page
- add these two lines at the end of it. Then press your escape key and ```:x```. You should exit the editing page
```
export LAB_DISCO_API_KEY=‘PUT IN YOUR API KEY HERE’
export CLOUDINARY_URL=‘cloudinary://851777568929886:GJN-qDx1C7idDTO4SZ92FuD3mI0@hqxlqewng’
```
- Do ```source .bashrc``` or ```source .vimrc``` to set these environment variables. You need to put in the class API key, change the value "export LAB_DISCO_API_KEY" is set to in the parentheses. 

5) Run the Application (in one separate terminal)
```bash
python3.12 app.py
```

5) Run redis (used for concurrent processing in heroku) (in one separate terminal)
```bash
brew update
brew install redis
redis-server
```
You may need to install redis on your computer if you haven't already. If you run into issues, check redis_config.py to ensure localhost url is set correctly.

Follow these instructions: https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

5.5) Access the redis cli to check the status of workers (optional)
```bash
pip install rq-dashboard
rq-dashboard
```
If you get SSL verification issues run: 
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org rq-dashboard
rq-dashboard
```


6) Run helper function in a separate terminal (in one separate terminal)
```bash
python3.12 worker.py
```


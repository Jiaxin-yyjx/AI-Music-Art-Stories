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

3) Set up your environment variables (only need to do this step once)
- in your terminal, type cd in an empty tab and then look for either a .bashrc or .zshrc file.
- Do ```cat .bashrc``` and ```cat .zshrc``` and whichever file has content in it is the file you want to modify.
- Do ```vim .bashrc``` or ```vim .zshrc``` accordingly
- press ```i``` and use your down arrow key to move to the bottom of the page
- add these two lines at the end of it. Then press your escape key and ```:x```. You should exit the editing page
```
export LAB_DISCO_API_KEY='PUT IN YOUR API KEY HERE'
export CLOUDINARY_URL='cloudinary://851777568929886:GJN-qDx1C7idDTO4SZ92FuD3mI0@hqxlqewng'
```
- Do ```source .bashrc``` or ```source .zshrc``` to set these environment variables. You need to put in the class API key, change the value "export LAB_DISCO_API_KEY" is set to in the parentheses. Follow the exact format.

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
You may need to install redis on your computer if you haven't already. If you run into issues, check redis_config.py to ensure localhost url is set correctly.brew update and brew install only need to be run the first time.

Follow these instructions: https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/

6) Run helper function in a separate terminal (in one separate terminal)
```bash
. .venv/bin/activate
python3.12 worker.py
```


7) Access the redis cli to check the status of workers (optional)
```bash
pip install rq-dashboard
rq-dashboard
```
If you get SSL verification issues run: 
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org rq-dashboard
rq-dashboard
```
Pip install only needs to be run once.

----------------------------------------------------------------
If you run into issues, ensure that you are using Python 3.12.8 specifically. Check using when your environment variable is activated.
```bash
python3.12 --version
```
If it does not return 3.12.8, download it using the link above. If you see a version of python other than 3.12, simply use python3.12 to run any commands. If you have another version of python3.12 like 3.12.2, follow these steps to change the default.

## Step 1: Locate the Python 3.12.8 Binary
To locate the Python 3.12.8 binary installed on your system, run the following command:

```bash
ls /usr/local/bin/python3.12
```
If the binary is correctly installed, you should see this output: ```/usr/local/bin/python3.12```

## Step 2: Update the PATH to Prioritize Python 3.12.8
1) Open your shell configuration file (e.g., ~/.zshrc for Zsh or ~/.bashrc for Bash): ```vim ~/.zshrc``` or ```vim ~/.bashrc```
2) Add the following line to the top of the file to prioritize Python 3.12.8: ```export PATH="/usr/local/bin:$PATH"```
3) Save and close the file by pressing the ```esc``` key and then typing ```:x``` and press enter.
4) Apply the changes: ```source ~/.zshrc``` or ```source ~/.bashrc```
5) Verify the correct version again ```python3.12 --version```

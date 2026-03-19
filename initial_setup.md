# clone the repo

# Step1: Install uv (if not already installed)
### Installion
Windows

- Copy paste below command into your powershell

Powershell 
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
If powershell is throwing execution policy error run below command and run uv installation command
```
Set-ExecutionPolicy RemoteSigned -scope CurrentUser
```
macOS/Linux:
Bash
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify if the uv is installed
```
uv --version
```
# Step2: Install git (if not installed)




# Step3: Clone

- Open your project folder and open git bash
- Run the below command
```
git clone (https://github.com/seeraseshasai-collab/apple-stock-analysis.git)
```


# Step3: install libraries
uv sync


## You are ready!

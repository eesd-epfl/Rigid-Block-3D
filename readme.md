# Set up Environment

Tested on **Ubuntu 20.04.6 LTS**.

## Step 1: Build Virtual Environment

```bash
python -m venv ./venv/
source ./venv/bin/activate
```

## Step 2: Configure Virtual Environment

```bash
cd path/to/setup.py
python setup.py develop
pip install -r requirements.txt
```

## Step 3: Obtain MOSEK License

Free academic and trial licenses are available at [MOSEK](https://www.mosek.com).


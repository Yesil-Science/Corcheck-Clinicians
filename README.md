# Corcheck-Clinicians
# FluiAI Django & Python3-Based Back-end 

This repo represents the new FluiAI app backend

## Getting Started

Please following the following steps in order to install the repo in your local/remote machines.

### Prerequisites

What things you need to install the software and how to install them

```
Python 3.7 (make sure it is added to your PC'c path)
```

* Fork/clone this repository using git
```
git clone https://github.com/Yesil-Science/FluAIBackend.git
```
* Then Create a new virtual environment inside this folder using the following commands

```
cd FluAIBackend

virtualenv venv
```

* We also need to activate this virtualenv using this:

```
venv\Scripts\activate
```

* For instuction about installing virtualenv package please refer to this We use [Link](https://www.geeksforgeeks.org/python-virtual-environment/).

### Installing

Install the project's packages using the following command

```
pip install -r requirements.txt
```

* After this step, make the required migrations to the db
```
python manage.py makemigrations

python manage.py migrate
```

* The final step is to run the backend server which can be easily done by the next instruction:

```
python manage.py runserver

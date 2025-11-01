# MLOps Equipo 63

### Conda/Pip Environment

Instalamos [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) y creamos un ambiente virtual (mna-mlops) para gestionar todas las librerias de Python con Pip.

```bash
conda --version
conda create -n mna-mlops python=3.12.0
conda activate mna-mlops
```

-----

### Cookiecutter Data Science

Una vez dentro del ambiente virtual de Conda, instalamos la librería de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) para estructurar el trabajo de acuerdo a los estándares de ciencia de datos:

```bash
pip install cookiecutter-data-science
ccds
```

Utilizamos la siguiente configuración para la estructura del proyecto:

```bash
project_name (project_name): mlops_equipo_63
repo_name (mlops_equipo_63): mlops_equipo_63
module_name (mlops_equipo_63): mlops_equipo_63
author_name (Your name (or your organization/company/team)): Equipo 63
description (A short description of the project.): Este proyecto tiene como proposito experimentar de manera practica como se construye, organiza y despliega un sistema de Machine Learning en producion, siguiendo los principios de MLOps.
python_version_number (3.10): 3.12.0
Select dataset_storage
    1 - none
    2 - azure
    3 - s3
    4 - gcs
    Choose from [1/2/3/4] (1): 1
Select environment_manager
    1 - virtualenv
    2 - conda
    3 - pipenv
    4 - uv
    5 - pixi
    6 - poetry
    7 - none
    Choose from [1/2/3/4/5/6/7] (1): 2
Select dependency_file
    1 - requirements.txt
    2 - pyproject.toml
    3 - environment.yml
    4 - Pipfile
    5 - pixi.toml
    Choose from [1/2/3/4/5] (1): 1
Select pydata_packages
    1 - none
    2 - basic
    Choose from [1/2] (1): 1
Select testing_framework
    1 - none
    2 - pytest
    3 - unittest
    Choose from [1/2/3] (1): 1
Select linting_and_formatting
    1 - ruff
    2 - flake8+black+isort
    Choose from [1/2] (1): 1
Select open_source_license
    1 - No license file
    2 - MIT
    3 - BSD-3-Clause
    Choose from [1/2/3] (1): 2
Select docs
    1 - mkdocs
    2 - none
    Choose from [1/2] (1): 2
Select include_code_scaffold
    1 - Yes
    2 - No
    Choose from [1/2] (1): 2
```

El resultado es un folder (mlops_equipo_63) con la siguiente estructura:

```txt
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         article-popularity and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── mlops_equipo_63   <- Source code for use in this project.
    │
    └── __init__.py             <- Makes article-popularity a Python module
```

-----

### Git/GitHub

Para gestionar el versionamiento del código, utilizamos [Git](https://git-scm.com/install) y lo vinculamos con nuestra cuenta de [GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github) para poder hacer cambios de manera local y empujarlos al repositorio remoto. Para esto un miembro del equipo creó el repositorio remoto [MLOps_Equipo_63](https://github.com/VelascoCode/MLOps_Equipo_63) (publico) y proporcionó permisos de lectura y escritura a los demás miembros del equipo.

Una vez que creada la estructura del proyecto con CoockieCutters, inicializamos Git en la raíz del proyecto (mlops_equipo_63), añadimos todos los archivos, creamos el commit, y empujamos todos los cambios a la rama principal del repositorio remoto:

```bash
git --version

cd mlops_equipo_63

git init
git add .
git commit -m "CCDS defaults"
git remote add origin https://github.com/VelascoCode/MLOps_Equipo_63
git branch -M main
git push -u origin main
```

-----

### Amazon Web Services (AWS): IAM User/Role, Access Key, S3 Bucket

Para poder almacenar las diversas versiones de los datos con los que vamos a estar trabajando, creamos un [Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) con la cuenta de AWS de un miembro del equipo: *s3://mlops-equipo-63*. En esta cuenta también creamos un [rol](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_s3_rw-bucket.html) con permisos de escritura y lectura hacía este S3 bucket. Este rol se asignó a diversos usuarios que también se crearon para que los demás miembros del equipo pudieran utilizar el S3 bucket a través de la linea de comando utilizando una [access key](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html). Por simplicidad, se utilizó la matrícula del TEC para definir el nombre de los usuarios. 

Para conectarnos a la cuenta de AWS con los usuarios previamente creados, instalamos [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) en el ambiente virtual de Conda y proporcionamos la access key correspondiente:

```bash
pip install awscli

aws --version

aws configure

AWS Access Key ID [****************5XMO]: 
AWS Secret Access Key [****************JlOz]:
Default region name [us-east-1]: us-east-1
Default output format [json]: json
```

Para validar la conexión con la cuenta de AWS utilizamos los siguentes comandos:

```bash
aws sts get-caller-identity
aws s3 ls
```

-----

### Data Version Control (DVC)

Para gestionar el versionamiento de los datos, utilizamos [DVC](https://dvc.org/doc/install) y la instancia de Amazon S3 bucket previamente creada. Instalamos DVC en el ambiente virtual de Conda y lo inicializamos en la raíz del proyecto:

```bash
pip install dvc

dvc --version

cd mlops_equipo_63

dvc init
git commit -m "Initialize DVC"
```

Establecemos la conexión entre DVC y el Amazon S3 bucket:

```bash
pip install dvc-s3

dvc remote add -d storage s3://mlops-equipo-63
```

Para validar el funcionamiento de DVC, creamos un archivo dummy en *data/raw/dummy.csv* y lo añadimos con DVC, creamos el commit correspondiente con Git y lo empujamos al S3 bucket con DVC:

```bash
mkdir -p data/raw && echo -e "id,name,age,city\n1,Alice,25,New York" > data/raw/dummy.csv # (v1)

dvc add data/raw/test.csv # un nuevo archivo se creará: data/raw/dummy.csv.dvc
git commit -m 'added dummy.csv.dvc file'
dvc push
```

Modificamos el archivo *dummy.csv*, revisamos el status con DVC y repetimos los pasos (add, commit, push):

```bash
echo "3,Charlie,35,Chicago" >> data/raw/dummy.csv # (v2)

dvc status

dvc add data/raw/test.csv
git commit -m 'modified dummy.csv.dvc file'
dvc push
```

Para regresar a la version 1 (v1), cambiamos de versión con Git y hacemos el pull de los datos con DVC:

```bash
git log --oneline
git checkout e5af546 # commit hash del commit 'added dummy.csv.dvc file'

dvc pull
```

Para regresar a la versión del último commit ejecutamos:

```bash
git checkout main 
dvc pull
```

-----

### MLflow

Para el versionamiento de los modelos de Machine Learning utilizamos MLflow. Instalamos MLflow en el ambiente virtual de Conda y lo ejecutamos dentro del folder */notebooks* para almacenar los experimentos en esta ruta:

```bash
pip install mlflow
mlflow server --host 127.0.0.1 --port 8080
```

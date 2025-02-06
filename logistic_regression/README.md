# Logistic Regression
Bienvenido a este proyecto desarrollado en Jupyter Lab con minicnoda instalado en un Linux basado en Ubuntu.

### Estoy en Linux Mint
sudo apt update -y
sudo apt upgrade -y

### Ver si está instalado conda en el sistema
conda --version

### Instalación de miniconda

### Instalar el entorno virtual llamado 'env'
conda create --name env

### Ir al nuevo entorno virtual
conda activate env

### Ver si está instalado jupyter lab
jupyter lab --version

### Instalación de jupyter lab
conda install -c conda-forge jupyterlab

### Verificar si se ha instalado jupyter lab y otros paquetes
#### Se puede ver la versión de Python instalada
conda list

### Lanzar jupyter lab estando en el entorno virtual 'env'
jupyter lab

### Ir al directorio de nuestro proyecto y hacer un git pull para actualizarlo
#### Lanzar jupyter lab. Se abrirá en el browser que tengamos
#### y en la terminal queda el servidor corriendo
jupyter lab

### Instalar pandas en 'env'. Asegurate de estar en 'env'
conda install pandas

### Instalar la librería seaborn desde el canal conda-forge
### este canal habitualmente tiene las librerías muy actualizadas
conda install -c conda-forge seaborn

### Instalar la librería tabulate
conda install -c conda-forge tabulate


## Crear un archivo environment.yml para capturar las librerías y versiones usadas en el entorno virutual
### Exportar el archivo environment.yml. Debo estar en el entorno virtual 'env'

conda env export > environment.yml

### Este comando creará un archivo environment.yml en tu directorio actual con todas las dependencias y versiones de tu entorno

### Verifica el contenido del archivo generado
cat environment.yml

## Para recrear el entorno a partir de este archivo, simplemente usa:
conda env create -f environment.yml

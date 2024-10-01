# Establece la imagen base
FROM php:8.1-apache

# Instala las dependencias necesarias, incluyendo CMake
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    python3-opencv \
    python3-venv \
    cmake 

# Configura y compila la extensión GD
RUN docker-php-ext-configure gd --with-jpeg \
    && docker-php-ext-install gd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crea un entorno virtual para Python
RUN python3 -m venv /opt/venv

# Instala face_recognition en el entorno virtual
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install face_recognition

# Establece la variable de entorno para usar el entorno virtual
ENV PATH="/opt/venv/bin:$PATH"

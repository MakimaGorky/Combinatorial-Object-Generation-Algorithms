"""
installation man

sage -pip install -e . # из корня пакета

"""

from setuptools import setup, find_packages

setup(
    name="combobgen",
    version="0.0.0.0",
    packages=find_packages(),
    description="из названия и так всё понятно 🤠",
    author="haha",
    python_requires='>=3.9',
    install_requires=[], # тут есть какие-то зависимости, но лень
)


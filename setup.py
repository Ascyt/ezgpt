from setuptools import setup, find_packages

setup(
        name='ezgpt',
        version='1.11.1',
        packages=find_packages(),
        description='A simple GPT interface',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        author='Filip Schauer',
        author_email='contact@ascyt.com',
        url='https://github.com/Ascyt/ezgpt',
        license='MIT',
        install_requires=[
            'openai>=1.0.0', 'colorama'
            ],
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License'
            ],
        python_requires='>=3.11.2'
        )

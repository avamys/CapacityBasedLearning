import os
from jupytext import cli


def is_jupyter(python_file: str) -> bool:
    with open(python_file, 'r') as file:
        for _ in range(3):
            if 'jupyter:\n' in file.readline():
                return True
    return False


def main():
    ''' Run to convert paired python files to jupyter notebooks '''
    for file in os.listdir():
        if file.endswith('.py'):
            if is_jupyter(file):
                cli.jupytext(['--set-formats', 'ipynb,py:percent', file])

if __name__ == '__main__':
    main()

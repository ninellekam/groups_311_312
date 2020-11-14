import os
import setuptools
import matrixgame


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
        name='matrix-game-project',
        version=matrixgame.__version__,
        author='Ivantsov-Satvaldina-Albatyrov',
        author_email='ivantsovgleb@icloud.com',
        #url='',
        description='A matrix-game package',
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        packages=setuptools.find_packages(),
        entry_points={
            'console_scripts':
                ['matrixgame = matrixgame.app:main']
            }
)



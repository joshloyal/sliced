from setuptools import setup


PACKAGES = [
    'rdlearn',
    'rdlearn.tests',
]


def setup_package():
    setup(
        name="reduce-learn",
        version='0.1.0',
        description='Scikit-Learn compatabile Sufficient Dimension Reduction algorithms.',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/reduce-learn',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES
    )


if __name__ == '__main__':
    setup_package()

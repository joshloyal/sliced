from setuptools import setup


PACKAGES = [
    'sir',
    'sir.tests',
]


def setup_package():
    setup(
        name="Sliced Inverse Regression",
        version='0.1.0',
        description='Scikit-Learn compatabile Sliced Inverse Regression',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/SlicedInverseRegression',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES
    )


if __name__ == '__main__':
    setup_package()

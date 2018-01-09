from setuptools import setup


PACKAGES = [
    'imreduce',
    'imreduce.tests',
]


def setup_package():
    setup(
        name="inverse-moment-reductions",
        version='0.1.0',
        description='Scikit-Learn compatabile Sufficient Dimension Reduction algorithms.',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/inverse-moment-reductions',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES
    )


if __name__ == '__main__':
    setup_package()

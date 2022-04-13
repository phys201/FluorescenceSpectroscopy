from setuptools import setup

setup(name='example',
      version='0.1',
      description='1',
      long_description=readme(),
      url='http://github.com/phys201/FluorescenceSpectroscopy',
      author='bdelwood, derickgonzalez, abdullahnasir-97',
      author_email='bdelwood@users.noreply.github.com',
      license='GPLv3',
      packages=['fluospec'],
      install_requires=['numpy',
                        'scipy',
                        'pymc3',
                        'pandas'])

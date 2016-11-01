from setuptools import setup

setup(name='deepseries',
      version='1.0',
      description='RNNs for time series',
      keywords='tensorflow, sequencing, recurrent networks',
      url='https://github.kdc.capitalone.com/ezw112/deepseries',
      author='Jesus Martinez-Manso',
      author_email='jesus.martinezmanso@capitalone.com',
      install_requires = ['numpy','pandas','scikit-learn','matplotlib'],
      packages=['deepseries'],
      package_data={},
      include_package_data=True,
      zip_safe=False)

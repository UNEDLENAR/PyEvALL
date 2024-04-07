from setuptools import setup

setup(
    name='PyEvALL',
    version='0.1.62',
    description='PyEvALL (The Python library to Evaluate ALL) is a evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWiDi (Learning with disagreement).',
    author='JORGE CARRILLO-DE-ALBORNOZ',
    author_email='jcalbornoz@lsi.uned.es',
    url="https://github.com/UNEDLENAR/PyEvALL/tree/main",
    include_package_data=True,
    packages=['pyevall','pyevall.metrics','pyevall.utils',
              'pyevall.comparators','pyevall.reports'],
    install_requires=['jsbeautifier==1.14.9', 'jsonschema==4.21.1', 'numpy==1.26.4', 'pandas==2.2.1', 'setuptools==68.0.0', 'tabulate==0.9.0'
    ],
)

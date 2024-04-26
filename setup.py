from setuptools import setup

setup(
    name='PyEvALL',
    version='0.1.63',
    description='PyEvALL (The Python library to Evaluate ALL) is a evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWiDi (Learning with disagreement).',
    long_description="""# PyEvALL\n PyEvALL (The Python library to Evaluate ALL) is an evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWiDi (Learning with disagreement). PyEvALL is designed based on the following concepts: (i) **persistence**, users can save evaluations and retrieve past evaluations; (ii) **replicability**, all evaluations are conducted using the same methodology, making them strictly comparable; (iii) **effectiveness**, all metrics are unified under measurement theory and have been doubly implemented and compared; (iv) **generalization**, achieved through the use of a standardized input format enabling users to evaluate all evaluation contexts.\n""",
    long_description_content_type='text/markdown',
    author='JORGE CARRILLO-DE-ALBORNOZ',
    author_email='jcalbornoz@lsi.uned.es',
    url="https://github.com/UNEDLENAR/PyEvALL/tree/main",
    include_package_data=True,
    packages=['pyevall','pyevall.metrics','pyevall.utils',
              'pyevall.comparators','pyevall.reports'],
    install_requires=['jsbeautifier==1.14.9', 'jsonschema==4.21.1', 'numpy==1.26.4', 'pandas==2.2.1', 'setuptools==68.0.0', 'tabulate==0.9.0'
    ],
)

'''
setup.py - a setup script
Author: Xinyu Wang
'''

from setuptools import setup, find_packages

import os
import sys
# import BGP_Forecast_Modules

try:
	setup(
		name='HCP_DataProcessor',
		version='0.0.4',
		author='Xinyu Wang',
		author_email='xinyuwang1209@gmail.com',
		description = ("HCP_DataProcessor."),
		url='https://github.com/xinyuwang1209/HCP_DataProcessor.git',
		platforms = 'any',
		classifiers=[
			'Environment :: Console',
			'Intended Audience :: Developers',
			'License :: OSI Approved :: MIT License',
			'Operating System :: OS Independent',
			'Programming Language :: Python',
			'Programming Language :: Python :: 3'
		],
		keywords=['Xinyu, xinyu, pypi, package, rpki'],
		packages=find_packages(include=['HCP_DataProcessor', 'HCP_DataProcessor.*']),
		install_requires=[
			'numpy',
			'pandas',
			'pathos',
			'sklearn'
		],

		)
finally:
	pass

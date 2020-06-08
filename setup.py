#!/usr/bin/env python
from setuptools import setup
import glob

templates = glob.glob("templates/*")

setup(name='acis_thermal_check',
      packages=["acis_thermal_check"],
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='ACIS Thermal Model Library',
      author='John ZuHone',
      zip_safe=False,
      platforms=['any'],
      author_email='john.zuhone@cfa.harvard.edu',
      url='http://github.com/acisops/acis_thermal_check',
      data_files=[('templates', templates)],
      include_package_data=True,
      )

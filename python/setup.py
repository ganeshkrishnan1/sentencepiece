#!/usr/bin/env python

# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.!

import codecs
import os
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

sys.path.append(os.path.join('.', 'test'))

# Set LevelDB paths
leveldb_include_dir = "C:/Program Files (x86)/leveldb/include"
leveldb_lib_dir = "C:/Program Files (x86)/leveldb/lib"

sentencepiece_include_dir="C:/Program Files (x86)/sentencepiece/include"
sentencepiece_lib_dir="C:/Program Files (x86)/sentencepiece/lib"

def long_description():
  with codecs.open('README.md', 'r', 'utf-8') as f:
    long_description = f.read()
  return long_description

exec(open('src/sentencepiece/_version.py').read())

def run_pkg_config(section, pkg_config_path=None):
  try:
    cmd = 'pkg-config sentencepiece --{}'.format(section)
    if pkg_config_path:
      cmd = 'env PKG_CONFIG_PATH={} {}'.format(pkg_config_path, cmd)
    output = subprocess.check_output(cmd, shell=True)
    if sys.version_info >= (3, 0, 0):
      output = output.decode('utf-8')
  except subprocess.CalledProcessError:
    sys.stderr.write('Failed to find sentencepiece pkg-config\n')
    sys.exit(1)
  return output.strip().split()

def is_sentencepiece_installed():
  try:
    subprocess.check_call('pkg-config sentencepiece --libs', shell=True)
    return True
  except subprocess.CalledProcessError:
    return False

def get_cflags_and_libs(root):
  cflags = ['-std=c++17', '-I' + os.path.join(root, 'include')]
  libs = ['-L' + os.path.join(root, 'lib'), '-lsentencepiece', '-lsentencepiece_train']
  return cflags, libs

if is_sentencepiece_installed():
  cflags, libs = run_pkg_config('cflags'), run_pkg_config('libs')
  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'],
      extra_compile_args=cflags,
      extra_link_args=libs,
  )
  cmdclass = {}
else:
  cflags = ['/std:c++17', f'/I{leveldb_include_dir}', '/I.\\build\\root\\include', f'/I{sentencepiece_include_dir}']
  libs = [
      f'{leveldb_lib_dir}/leveldb.lib',
      f'{sentencepiece_lib_dir}/sentencepiece.lib',
      f'{sentencepiece_lib_dir}/sentencepiece_train.lib',
  ]

  SENTENCEPIECE_EXT = Extension(
      'sentencepiece._sentencepiece',
      sources=['src/sentencepiece/sentencepiece_wrap.cxx'],
      extra_compile_args=cflags,
      extra_link_args=libs,
      include_dirs=[sentencepiece_include_dir, leveldb_include_dir, '.\\build\\root\\include'],
      libraries=['leveldb'],
      library_dirs=[leveldb_lib_dir],
  )
  cmdclass = {'build_ext': _build_ext}

setup(
    name='sentencepiece',
    author='Taku Kudo',
    author_email='taku@google.com',
    description='SentencePiece python wrapper',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    version=__version__,
    package_dir={'': 'src'},
    url='https://github.com/google/sentencepiece',
    license='Apache',
    platforms='Unix',
    py_modules=[
        'sentencepiece/__init__',
        'sentencepiece/_version',
        'sentencepiece/sentencepiece_model_pb2',
        'sentencepiece/sentencepiece_pb2',
    ],
    ext_modules=[SENTENCEPIECE_EXT],
    cmdclass=cmdclass,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    test_suite='sentencepiece_test.suite',
)

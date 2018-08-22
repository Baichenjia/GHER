from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='GHER',
      packages=[package for package in find_packages()
                if package.startswith('GHER')],
      install_requires=[
          'gym[mujoco,robotics]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'progressbar2',
          'mpi4py',
          'cloudpickle',
          'tensorflow-gpu>=1.8.0',
          'baselines',
          'click',
      ],
      description='Guided goal generation for multi-goal reinforcement learning.',
      author='ChenjiaBai',
      url='https://github.com/Baichenjia/GHER/',
      author_email='bai_chenjia@stu.hit.edu.cn',
      version='0.0.1')

import subprocess
import platform
import os
import argparse


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(THIS_DIR, os.pardir)
BUILD_DIR = os.path.join(ROOT_DIR, '_build')
DIST_DIR = os.path.join(ROOT_DIR, 'dist')

def main():
    parser = argparse.ArgumentParser(description='Build and release script')
    parser.add_argument('-p', '--platform', default='Windows7_x64_VS2015', help='target platform e.g: Windows7_x86_VS2013')
    args = parser.parse_args()

    return subprocess.call(['cmake',
                            '-G', 'Visual Studio 15 2017 Win64',
                            '-S', ROOT_DIR,
                            '-B', BUILD_DIR])


if __name__ == '__main__':
    import sys
    sys.exit(main())

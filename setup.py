import setuptools
with open("README.md","rt") as rm:
  long_desc=rm.read()
setuptools.setup(
    name='tf_tensor_dumper',
    version='0.0.1',
    description='Simple tensor dumper for Tensorflow for debugging',
    url='http://github.com/samikama/tf_tensor_dumper',
    author='Sami Kama',
    author_email='tf_tensor_dumper@samikama.com',
    license='Apache-2.0',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Utilities",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
    ],
    )

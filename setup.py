from setuptools import setup, find_namespace_packages
import os.path as op
import os

def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy/MNE-Python
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)

if __name__ == "__main__":
    setup(
          name='gssc',
          python_requires=">=3.7",
          maintainer='Jevri Hanna',
          include_package_data=True,
          maintainer_email="jevri.hanna@gmail.com",
          description="Greifswald Sleep Stage Classifier for automatic "
                      "detection of sleep stages in polysomnographies.",
          license="GNU Affero General Public License",
          url="https://github.com/jshanna100/gssc",
          version='0.0.3',
          classifiers=["Intended Audience :: Science/Research",
                       "License :: OSI Approved",
                       "Programming Language :: Python :: 3",
                       "Topic :: Scientific/Engineering",
                       "Operating System :: Unix",
                       "Operating System :: Microsoft :: Windows",
                       "Operating System :: MacOS"],
          install_requires=['mne', 'torch', 'importlib-resources'],
          packages=package_tree("gssc"),
          package_data={"gssc": [op.join("nets", "*.pt")]},
          entry_points={
            "console_scripts":["gssc_infer = gssc.command.com_infer:main"]
          }
    )

from pathlib import Path

from setuptools import find_packages, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise RuntimeError(
        "Cython is required to build .so extensions. "
        "Install it first with: pip install cython"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
SOURCE_PATTERNS = [
    "mef/*.py",
    "pipeline/*.py",
    "service/*.py",
]

EXCLUDE_PATTERNS = [
    "mef/__init__.py",
    "pipeline/__init__.py",
    "service/__init__.py",
]


setup(
    name="codeup-mef-hdr",
    version="0.1.0",
    description="MEF HDR image fusion service and worker pipeline",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "",
    long_description_content_type="text/markdown",
    author="qzq",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["main"],
    ext_modules=cythonize(
        SOURCE_PATTERNS,
        exclude=EXCLUDE_PATTERNS,
        compiler_directives={
            "language_level": "3",
            "annotation_typing": False,
        },
    ),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "fastapi",
        "loguru",
        "numpy",
        "opencv-python",
        "Pillow",
        "python-multipart",
        "pyzmq",
        "toml",
        "torch",
        "torchvision",
        "traits",
        "uvicorn",
    ],
    entry_points={
        "console_scripts": [
            "mef-hdr=main:main",
        ],
    },
    data_files=[
        ("config", ["config/config.toml"]),
        ("weights", ["weights/weight.pth"]),
    ],
)

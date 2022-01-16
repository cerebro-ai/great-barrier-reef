import setuptools

setuptools.setup(
    name="gbr",
    version="0.0.2",
    author="Cerebro AI",
    description="Code for the kaggle challenge 'Great Barrier Reef'",
    packages=setuptools.find_packages(include=['gbr', 'gbr.*']),
    install_requires=[
        "torch~=1.10",
        "torchvision~=0.11.1",
        "pandas",
        "pillow~=8.4.0",
        "numpy",
        "pyyaml",
        "wandb",
        "moviepy",
        "imageio",
        "albumentations",
        "randomname",
        "mmcv",
        "easydict"
    ]
)

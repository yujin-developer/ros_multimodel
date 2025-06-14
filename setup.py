from setuptools import find_packages, setup

package_name = 'multimodel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            f'lib/python3.10/site-packages/{package_name}', 
            ['multimodel/fruits_weight_sphercity.csv']
        ),
        (
            f'lib/python3.10/site-packages/{package_name}', 
            ['multimodel/Breast_Cancer.csv']
        ),
        (
            f'lib/python3.10/site-packages/{package_name}', 
            ['multimodel/Iris.csv']
        ),
        (
            f'lib/python3.10/site-packages/{package_name}', 
            ['multimodel/Penguin.csv']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yujin',
    maintainer_email='yujin.jeon@ue-germany.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fruit_publisher = multimodel.fruit_publisher:main',
            'iris_publisher = multimodel.iris_publisher:main',
            'svm_publisher = multimodel.svm_publisher:main',
            'fruit_subscriber = multimodel.fruit_subscriber:main',
            'iris_subscriber = multimodel.iris_subscriber:main',
            'svm_subscriber = multimodel.svm_subscriber:main',
            'model_performance_subscriber = multimodel.model_performance_subscriber:main'
        ],
    },
)

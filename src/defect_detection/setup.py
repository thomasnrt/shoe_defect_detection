from setuptools import find_packages, setup

package_name = 'defect_detection'

setup(
    name=package_name,
    version='0.0.0',
    # Indique à setuptools de trouver ton sous-dossier defect_detection
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lila',
    maintainer_email='lila@todo.todo',
    description='Package pour la détection de défauts sur chaussures via ResNet-18 et U-Net',
    license='TODO: License declaration',
    tests_require=['pytest'],
entry_points={
        'console_scripts': [
            # --- CEUX QUE TU VAS UTILISER LÀ MAINTENANT ---
            # Exécutable 'node_dd' -> lance la fonction main() de ton fichier node_dd.py
            'node_dd = defect_detection.node_dd:main',
            
            # Exécutable 'node_photo' -> lance la fonction main() de ton fichier node_photo.py
            'node_photo = defect_detection.node_photo:main',
            
            # --- LES AUTRES SCRIPTS AU CAS OÙ ---
            # Exécutable 'node_ia' -> lance ton fichier node_def_dect.py
            'node_ia = defect_detection.node_def_dect:main',
            
            # Exécutable 'simulateur_camera' -> lance ton fichier simulateur_camera.py
            'simulateur_camera = defect_detection.simulateur_camera:main',
        ],
    },
)

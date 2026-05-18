from setuptools import find_packages, setup

package_name = "dmz_sentry_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rokey",
    maintainer_email="rokey@example.com",
    description="Mission and patrol control nodes for the DMZ Sentry Isaac Sim project.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "patrol_controller = dmz_sentry_control.patrol_controller:main",
        ],
    },
)

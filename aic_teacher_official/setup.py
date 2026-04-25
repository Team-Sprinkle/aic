from setuptools import find_packages, setup

package_name = "aic_teacher_official"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team Sprinkle",
    maintainer_email="team-sprinkle@example.com",
    description="Official teacher trajectory replay pipeline for AIC.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
)

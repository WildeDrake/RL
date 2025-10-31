import pkg_resources

installed_packages = pkg_resources.working_set
for package in sorted(installed_packages, key=lambda x: x.key):
    print(f"{package.key}=={package.version}")

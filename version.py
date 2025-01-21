from pdm.backend.hooks.version import SCMVersion

def format_version(version: SCMVersion) -> str:
    print("welcome in my formatter")
    print({str(version.version)})
    print({str(version.distance)})
    if version.distance is None:
        return str(version.version)
    else:
        return f"{version.version}.post{version.distance}"
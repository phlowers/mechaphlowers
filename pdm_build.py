import os

from pdm.backend.hooks.version import SCMVersion  # type: ignore


def format_version(version: SCMVersion) -> str:
    """Function provided for pdm backend to customize the version format when extract from scm tag.

    In practice, this function is called by the pdm backend to format the version string.
    When called with the PACKAGE_BUILD_TEST environment variable set, it will append a
    .dev{PACKAGE_BUILD_TEST} to the version string. This is useful for testing deployment ci.

    Note that this function is not called if the PDM_BUILD_SCM_VERSION environment variable is set.
    In this case there is no control from this function but only from pdm backend.
    """
    print("---- pdm custom build version ----")
    print(
        f"INFO: Version tag: {version.version}",
    )
    test_build = os.getenv("PACKAGE_BUILD_TEST", "")

    dev_str = ""
    if test_build != "":
        dev_str = f".dev{test_build}"
        print(f"INFO Test build is activated, append {dev_str} to version")
        return str(version.version) + dev_str

    print("INFO Final version build")
    if version.distance is None:
        return str(version.version)
    else:
        print(
            f"INFO Version is a post release, append .post{version.distance} to version"
        )
        return f"{version.version}.post{version.distance}"

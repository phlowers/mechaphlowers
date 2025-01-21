from pdm.backend.hooks.version import SCMVersion
import os

def format_version(version: SCMVersion) -> str:
    """format_version : function provided for pdm backend to customize the version format when extract from scm tag.
    In practice, this function is called by the pdm backend to format the version string
    When called with the PACKAGE_BUILD_TEST environment variable set, it will append a 
    .dev{PACKAGE_BUILD_TEST} to the version string. This is useful for testing deployment ci.
    
    Note that this function is not called if the PDM_BUILD_SCM_VERSION environment variable is set.
    In this case there is no control from this function but only from pdm backend.
    """
    
    test_build = os.getenv("PACKAGE_BUILD_TEST","")
    
    dev_str = ""
    if test_build != "":
        dev_str = f".dev{test_build}"
    
    if version.distance is None:
        return str(version.version)+dev_str
    else:
        return f"{version.version}.post{version.distance}"+dev_str
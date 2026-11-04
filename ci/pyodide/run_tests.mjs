// Install the built mechaphlowers wheel in Pyodide and run the test suite.
// Usage: node run_tests.mjs <wheel> <tests_dir> <constraints_file> [pytest args...]
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";

const [wheelPath, testsDir, constraintsPath, ...pytestArgs] =
  process.argv.slice(2);
if (!wheelPath || !testsDir || !constraintsPath) {
  console.error(
    "Usage: node run_tests.mjs <wheel> <tests_dir> <constraints_file> [pytest args...]",
  );
  process.exit(2);
}

// Same minimum as the "test" dependency group of pyproject.toml.
const TEST_REQUIREMENTS = ["requests>=2.34.0"];
// Packages shipped by Pyodide (compiled or pre-built), loaded instead of PyPI.
const PYODIDE_PACKAGES = [
  "micropip",
  "numpy",
  "pandas",
  "scipy",
  "pyproj",
  "pydantic",
  "pyyaml",
  "xxhash",
  "pytest",
];

const constraints = fs
  .readFileSync(constraintsPath, "utf8")
  .split("\n")
  .map((line) => line.split("#")[0].trim())
  .filter(Boolean);

function copyDir(pyodide, src, dest) {
  pyodide.FS.mkdirTree(dest);
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (entry.name === "__pycache__" || entry.name === "benchmark") continue;
    const from = path.join(src, entry.name);
    const to = `${dest}/${entry.name}`;
    if (entry.isDirectory()) {
      copyDir(pyodide, from, to);
    } else if (entry.isFile()) {
      pyodide.FS.writeFile(to, fs.readFileSync(from));
    }
  }
}

const pyodide = await loadPyodide();
console.log(`Pyodide ${pyodide.version}, Python ${pyodide.runPython("import sys; sys.version")}`);

const errors = [];
await pyodide.loadPackage(PYODIDE_PACKAGES, {
  errorCallback: (message) => errors.push(message),
});
const missing = PYODIDE_PACKAGES.filter(
  (name) => !(name in pyodide.loadedPackages),
);
if (missing.length > 0) {
  console.error(`Failed to load Pyodide packages: ${missing.join(", ")}`);
  console.error(errors.join("\n"));
  console.error(
    "Check network access to cdn.jsdelivr.net. Behind an HTTP proxy, set NODE_USE_ENV_PROXY=1.",
  );
  process.exit(1);
}

const wheelName = path.basename(wheelPath);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheelName}`, fs.readFileSync(wheelPath));
copyDir(pyodide, testsDir, "/work/test");

pyodide.globals.set("wheel_requirement", `mechaphlowers[full] @ emfs:/wheels/${wheelName}`);
pyodide.globals.set("test_requirements", pyodide.toPy(TEST_REQUIREMENTS));
pyodide.globals.set("constraints", pyodide.toPy(constraints));
pyodide.globals.set("pytest_args", pyodide.toPy(pytestArgs));

const exitCode = await pyodide.runPythonAsync(`
import os
import micropip

# certifi is already loaded by Pyodide with an older version: force the
# security minimum from the constraints.
await micropip.install(
    [c for c in constraints if c.startswith("certifi")], reinstall=True
)
await micropip.install(
    [wheel_requirement, *test_requirements],
    constraints=list(constraints),
)

# Run from a directory without pyproject.toml, so that the installed wheel is
# tested rather than the sources.
os.chdir("/work")
import pytest

int(pytest.main(["test", "-p", "no:cacheprovider", *pytest_args]))
`);

process.exit(exitCode);

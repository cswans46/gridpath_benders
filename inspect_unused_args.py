from pathlib import Path
import ast
import pandas as pd


def get_function_parameter_usage(file_path):
    """Return parameter usage information for every function in a Python file."""

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # All parameters
        parameters = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        )

        if node.args.vararg:
            parameters.append(node.args.vararg)

        if node.args.kwarg:
            parameters.append(node.args.kwarg)

        parameter_names = {arg.arg for arg in parameters}

        # Names actually read/used in the function
        used_names = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
        }

        used_parameters = parameter_names & used_names
        unused_parameters = parameter_names - used_parameters

        results.append({
            "function": node.name,
            "line": node.lineno,
            "parameters": sorted(parameter_names),
            "used_parameters": sorted(used_parameters),
            "unused_parameters": sorted(unused_parameters),
        })

    return results


def inspect_directory(directory):
    """Inspect all Python functions recursively in a directory."""

    directory = Path(directory)

    results = []

    for file_path in directory.rglob("*.py"):
        try:
            functions = get_function_parameter_usage(file_path)

            for function in functions:
                function["file"] = str(file_path)
                results.append(function)

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Could not parse {file_path}: {e}")

    return results

project_dir = Path(__file__).resolve().parent / "gridpath"

results = inspect_directory(project_dir)

results = pd.DataFrame(results)

unused = results[
    results["used_parameters"].apply(
        lambda x: "subproblem" in x
    )
    &
    results["function"].apply(
        lambda x: "add_model_components" in x
    )
]

print(
    unused[
        "file"
    ].drop_duplicates().map(lambda x: Path(x).name).to_string(index=False)
)

unused["file"].drop_duplicates().map(lambda x: Path(x).name).to_csv(
    "unused_files.csv",
    index=False,
    header=["file"]
)



# for result in results:
#     print(f"\n{result['file']}:{result['line']}")
#     print(f"  Function: {result['function']}")
#     print(f"  Parameters: {result['parameters']}")
#     print(f"  Used: {result['used_parameters']}")
#     print(f"  Unused: {result['unused_parameters']}")